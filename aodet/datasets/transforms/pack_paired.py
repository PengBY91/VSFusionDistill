"""Degradation sampling, teacher/student fork and packing (§4.4(3)).

The fork point is fixed by the setup doc: geometric augmentation first,
copy the clean camera image for the teacher, then degrade the student's
copy.  ``AOStudentCameraView`` implements exactly that order dependency:
it must come after all geometric transforms in the pipeline.

D∞ "drop": ``img_camera`` is set to None (the preprocessor forwards None
and the detector skips the camera stream).  The teacher still receives the
clear copy, so distillation keeps working at the D∞ endpoint.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample

from .camera_degradation import ALL_LEVELS, degrade_camera

__all__ = [
    "AOSampleDegLevel",
    "AOStudentCameraView",
    "AOPackPairedInputs",
]


@TRANSFORMS.register_module()
class AOSampleDegLevel(BaseTransform):
    """Draw the training degradation level from p(s) (F2 §5.2).

    Defaults are the F2 §5.2 initial values: D0–D4 each 0.16, D∞ 0.2 —
    frozen on the validation set before real runs.
    """

    _DEFAULT_PROBS = {"d0": 0.16, "d1": 0.16, "d2": 0.16, "d3": 0.16,
                      "d4": 0.16, "dinf": 0.20}

    def __init__(self, probs: Optional[dict] = None, seed: int = 0):
        super().__init__()
        p = dict(self._DEFAULT_PROBS)
        if probs:
            p.update(probs)
        total = sum(p.values())
        assert abs(total - 1.0) < 1e-6, f"probs must sum to 1, got {total}"
        self.levels = list(p)
        self.probs = np.asarray([p[k] for k in self.levels])
        self.rng = np.random.default_rng(seed)

    def transform(self, results: dict) -> dict:
        results["deg_level"] = str(self.rng.choice(self.levels, p=self.probs))
        # per-sample draw seed: stays fixed per item so D0 and degraded
        # draws of the same item pair up deterministically in analysis
        results["deg_seed"] = int(results.get("deg_seed", 0))
        return results


@TRANSFORMS.register_module()
class AOStudentCameraView(BaseTransform):
    """Fork the camera image: clear copy for the teacher, degraded for the
    student.  Must run after all geometric transforms (§4.4(3))."""

    def __init__(self, dinf_mode: str = "drop", jitter: float = 0.08):
        super().__init__()
        assert dinf_mode in ("drop", "zero")
        self.dinf_mode = dinf_mode
        self.jitter = jitter

    def transform(self, results: dict) -> dict:
        img = results.get("img_camera")
        if img is None:
            return results
        level = results.get("deg_level", "d0")
        assert level in ALL_LEVELS
        results["img_camera_clear"] = img.copy()  # teacher view, clean
        if level == "dinf" and self.dinf_mode == "drop":
            results["img_camera"] = None
            return results
        results["img_camera"] = degrade_camera(
            img, sample_id=results["sample_id"], level=level,
            seed=results.get("deg_seed", 0), jitter=self.jitter)
        return results


@TRANSFORMS.register_module()
class AOPackPairedInputs(BaseTransform):
    """Pack both sides into CHW tensors + DetDataSample lists."""

    def _pack_side(self, results: dict, name: str) -> tuple:
        img_key = f"img_{name}"
        img = results.get(img_key)
        boxes = np.asarray(results.get(f"boxes_{name}") or [],
                           dtype=np.float32)
        labels = np.asarray(results.get(f"labels_{name}") or [],
                            dtype=np.int64)
        if img is None:
            return None, None
        ds = DetDataSample()
        h, w = img.shape[:2]
        ori = results.get(f"ori_shape_{name}", (h, w))
        ds.set_metainfo(dict(
            sample_id=str(results["sample_id"]),
            img_shape=(h, w),
            ori_shape=(int(ori[0]), int(ori[1])),
            scale_factor=(w / ori[1], h / ori[0]),
            # COCO image id on this side (from frame_meta.csv) so saved
            # predictions map straight back to the GT json
            coco_image_id=results.get(f"coco_{name}_image_id"),
        ))
        gt = InstanceData()
        if len(boxes):
            b = boxes.copy()
            b[:, 2:] = b[:, :2] + b[:, 2:]  # xywh -> xyxy
            gt.bboxes = torch.from_numpy(b)
        else:
            gt.bboxes = torch.zeros((0, 4), dtype=torch.float32)
        gt.labels = torch.from_numpy(labels.astype(np.int64))
        ds.gt_instances = gt
        tensor = np.ascontiguousarray(img.transpose(2, 0, 1))
        return tensor, ds

    def transform(self, results: dict) -> dict:
        packed = dict(
            sample_id=str(results["sample_id"]),
            deg_level=results.get("deg_level", "d0"),
            deg_seed=int(results.get("deg_seed", 0)),
            flip_applied=bool(results.get("flip_applied", False)),
        )
        for name in ("sonar", "camera"):
            img, ds = self._pack_side(results, name)
            packed[f"img_{name}"] = img
            packed[f"data_samples_{name}"] = ds
        clear = results.get("img_camera_clear")
        packed["img_camera_clear"] = (
            np.ascontiguousarray(clear.transpose(2, 0, 1))
            if clear is not None else None)
        return packed
