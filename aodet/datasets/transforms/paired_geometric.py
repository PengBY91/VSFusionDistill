"""Geometric transforms for the two unregistered image planes (§4.4(2)).

Rules implemented here:
* **Scaling/cropping is per-side** — the two images do not share a
  projection, so their geometry may be adjusted independently.
* **Horizontal flips are synchronized or absent**: a camera flip mirrors
  the sonar bearing, which is only meaningful when the sonar image is 0°-
  centred (``sonar_zero_centred`` from the dataset metadata).  Otherwise
  ``AOPairedSyncFlip`` is a no-op and records that flips were skipped.
* No Mosaic/MixUp anywhere (they break pairing and teacher caching).

All transforms register with mmengine TRANSFORMS and operate on the plain
dict produced by PairedSonarCameraDataset.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np
from mmengine.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

__all__ = [
    "AOLoadPaired",
    "AOPairedResize",
    "AOPairedSyncFlip",
    "AOSonarScaleJitter",
]


@TRANSFORMS.register_module()
class AOLoadPaired(BaseTransform):
    """Read img_camera / img_sonar from fp_camera / fp_sonar (BGR, HWC)."""

    def transform(self, results: dict) -> dict:
        for fp_key, img_key in (("fp_camera", "img_camera"),
                                ("fp_sonar", "img_sonar")):
            fp = results.get(fp_key)
            if fp and os.path.exists(fp):
                img = cv2.imread(fp, cv2.IMREAD_COLOR)
                if img is None:
                    raise IOError(f"failed to read {fp}")
                results[img_key] = img
                results[f"ori_shape_{img_key[4:]}"] = tuple(
                    int(v) for v in img.shape[:2])
            else:
                results[img_key] = None
        return results


def _scale_boxes_xywh(boxes: list, fx: float, fy: float) -> list:
    return [[b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy] for b in boxes]


@TRANSFORMS.register_module()
class AOPairedResize(BaseTransform):
    """Resize each side to its own (w, h); boxes scale per-side."""

    def __init__(self, camera_size: Optional[tuple] = None,
                 sonar_size: Optional[tuple] = None,
                 keep_ratio: bool = False):
        super().__init__()
        if not camera_size and not sonar_size:
            raise ValueError("at least one of camera_size/sonar_size")
        self.camera_size = camera_size
        self.sonar_size = sonar_size
        self.keep_ratio = keep_ratio

    def _resize_side(self, img: Optional[np.ndarray], size: tuple,
                     boxes: list) -> tuple:
        if img is None or size is None:
            return img, boxes
        h0, w0 = img.shape[:2]
        if self.keep_ratio:
            scale = min(size[0] / w0, size[1] / h0)
            new = (int(w0 * scale + 0.5), int(h0 * scale + 0.5))
        else:
            new = (int(size[0]), int(size[1]))
        img = cv2.resize(img, new, interpolation=cv2.INTER_LINEAR)
        fx, fy = new[0] / w0, new[1] / h0
        return img, _scale_boxes_xywh(boxes, fx, fy)

    def transform(self, results: dict) -> dict:
        img, boxes = self._resize_side(
            results.get("img_camera"), self.camera_size,
            results.get("boxes_camera") or [])
        results["img_camera"], results["boxes_camera"] = img, boxes
        img, boxes = self._resize_side(
            results.get("img_sonar"), self.sonar_size,
            results.get("boxes_sonar") or [])
        results["img_sonar"], results["boxes_sonar"] = img, boxes
        return results


@TRANSFORMS.register_module()
class AOPairedSyncFlip(BaseTransform):
    """Synchronized horizontal flip — camera and sonar together or not at all.

    Only active when ``sonar_zero_centred`` (dataset metadata); records
    ``flip_applied`` so the M₀ consumer can mirror its map (see
    aodet.geometry.m0.mirror_m0).
    """

    def __init__(self, prob: float = 0.5, allow: bool = True):
        super().__init__()
        self.prob = prob
        self.allow = allow

    def transform(self, results: dict) -> dict:
        results["flip_applied"] = False
        if not self.allow or self.prob <= 0:
            return results
        if not results.get("sonar_zero_centred", False):
            # geometric precondition failed: never flip (§4.4(2))
            return results
        if np.random.rand() > self.prob:
            return results

        for img_key, boxes_key in (("img_camera", "boxes_camera"),
                                   ("img_sonar", "boxes_sonar")):
            img = results.get(img_key)
            if img is None:
                continue
            h, w = img.shape[:2]
            results[img_key] = img[:, ::-1, :].copy()
            results[boxes_key] = [
                [w - b[0] - b[2], b[1], b[2], b[3]]
                for b in (results.get(boxes_key) or [])
            ]
        results["flip_applied"] = True
        return results


@TRANSFORMS.register_module()
class AOSonarScaleJitter(BaseTransform):
    """Randomly rescale the sonar range axis (rows) on a FIXED canvas.

    E07/E08 motivation: apparent echo height drifts across recordings
    (100927 echoes are 12-17 px against a 22-39 px train support), and
    the box regression locks onto the train scale prior — localisation
    transfers, scale does not.  Jittering the row axis during training
    broadens that prior.

    **Must run after** ``AOPairedResize``: a fixed-size resize placed
    after this transform rescales every image back to the same row count
    and undoes the jitter exactly (review 2026-10-31 §1.2 — the first E08
    config had this order and was a no-op).  The canvas keeps its (H, W):
    f > 1 crops an H-row window (preferring one that keeps every box),
    f < 1 pads with ``pad_val``.  Boxes are shifted and clipped; boxes
    thinner than ``min_box_h`` after clipping are dropped with their
    labels.  The azimuth (column) axis, the camera stream and the camera
    GT stay fixed: cross-modal geometric consumers (M0 bearing windows)
    live in column space, which this transform does not touch.  Train-only.

    Args:
        scale_range: range of the row rescale factor f, drawn uniformly;
            (1, 1) disables the jitter.
        prob: probability of applying the jitter at all.
        pad_val: fill value for rows exposed when f < 1.
        min_box_h: boxes shorter than this after clipping are dropped.
    """

    def __init__(self, scale_range: tuple = (0.6, 1.8), prob: float = 1.0,
                 pad_val: int = 0, min_box_h: float = 2.0):
        super().__init__()
        if not 0 < scale_range[0] <= scale_range[1]:
            raise ValueError(f"invalid scale_range {scale_range}")
        self.scale_range = tuple(scale_range)
        self.prob = prob
        self.pad_val = pad_val
        self.min_box_h = min_box_h

    def transform(self, results: dict) -> dict:
        img = results.get("img_sonar")
        results["sonar_scale_jitter"] = 1.0
        if (img is None or self.prob <= 0 or self.scale_range == (1, 1)
                or np.random.rand() > self.prob):
            return results
        f = float(np.random.uniform(*self.scale_range))
        h, w = img.shape[:2]
        new_h = max(1, int(round(h * f)))
        if new_h == h:
            return results
        scaled = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_LINEAR)
        fy = new_h / h
        boxes = [[b[0], b[1] * fy, b[2], b[3] * fy]
                 for b in (results.get("boxes_sonar") or [])]

        if new_h > h:
            # crop an h-row window; keep every box when that is possible
            lo, hi = 0, new_h - h
            if boxes:
                top = min(b[1] for b in boxes)
                bot = max(b[1] + b[3] for b in boxes)
                k_lo = max(0, int(np.ceil(bot - h)))
                k_hi = min(new_h - h, int(np.floor(top)))
                if k_lo <= k_hi:
                    lo, hi = k_lo, k_hi
            y0 = int(np.random.randint(lo, hi + 1))
            out = np.ascontiguousarray(scaled[y0:y0 + h])
            shift = -y0
        else:
            off = int(np.random.randint(0, h - new_h + 1))
            out = np.full_like(img, self.pad_val)
            out[off:off + new_h] = scaled
            shift = off

        labels = list(results.get("labels_sonar") or [])
        kept_boxes, kept_labels = [], []
        for i, b in enumerate(boxes):
            y1 = max(b[1] + shift, 0.0)
            y2 = min(b[1] + b[3] + shift, float(h))
            if y2 - y1 >= self.min_box_h:
                kept_boxes.append([b[0], y1, b[2], y2 - y1])
                if i < len(labels):
                    kept_labels.append(labels[i])
        results["img_sonar"] = out
        results["boxes_sonar"] = kept_boxes
        if labels:
            results["labels_sonar"] = kept_labels
        results["sonar_scale_jitter"] = f
        return results
