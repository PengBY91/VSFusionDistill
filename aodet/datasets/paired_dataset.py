"""PairedSonarCameraDataset: mmengine BaseDataset over the paired format.

Reading follows aodet.datasets.paired_format (setup doc §4.4(1)): every
source dataset was converted once into camera.json / sonar.json / pairs.csv
/ frame_meta.csv; this dataset only reads that layout.

Pipeline contract (transforms receive a plain dict):

    img_camera (HWC BGR), img_sonar (HWC BGR), boxes_camera, boxes_sonar,
    labels_camera, labels_sonar, sample_id, meta (frame_meta row)

The recommended order (setup doc §4.4(3) — geometric augmentation first,
then the teacher/student fork, then degradation):

    LoadPairedImages → AOPairedResize(+sync flip, usually off) →
    AOSampleDegLevel (train only) → AOStudentCameraView (fork + degrade) →
    AOPackPairedInputs

Evaluation loaders skip AOSampleDegLevel and pass a fixed ``deg_level``
override (the offline test sets of §4.4(3) are the recommended path; the
override exists for quick scans).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from mmdet.registry import DATASETS
from mmdet.structures import DetDataSample
from mmengine.dataset import BaseDataset
from mmengine.structures import InstanceData

from .paired_format import read_paired

__all__ = ["PairedSonarCameraDataset"]


@DATASETS.register_module()
class PairedSonarCameraDataset(BaseDataset):
    """Two images, two annotation sets, one sample_id per item.

    Args:
        data_root: paired-format root (contains camera.json etc. and
            images/).
        split: value matched against frame_meta.csv's ``split`` column
            (e.g. 'train', 'val', 'test').
        camera_off: drop the camera stream at load time (B0 mode reads the
            sonar half only).
        deg_level: fixed degradation level for this loader (default: taken
            from frame_meta ``deg_level`` column or 'd0'); the
            AOSampleDegLevel transform overrides it per-sample for training.
        sonar_zero_centred: whether the sonar image bearing axis is 0°-
            centred (sync flips are only legal when True, §4.4(2)).
        sonar_gt_sources: which sonar annotation ``source`` values to
            train on ("blob" = measured echo, "projected" = calibrated
            geometry).  Default keeps both; restrict to ("blob",) when
            the projected boxes' calibration error is not yet
            acceptable -- frames left without sonar boxes become
            negative samples, so this trades label noise for occasional
            false negatives.
    """

    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 camera_off: bool = False,
                 deg_level: Optional[str] = None,
                 sonar_zero_centred: bool = True,
                 sonar_gt_sources: Sequence[str] = ("blob", "projected"),
                 pipeline: Optional[list] = None,
                 **kwargs):
        self.split = split
        self.camera_off = camera_off
        self.fixed_deg_level = deg_level
        self.sonar_zero_centred = sonar_zero_centred
        self.sonar_gt_sources = tuple(sonar_gt_sources)
        # BaseDataset will call load_data_list; data_root must be set first
        kwargs.setdefault("data_prefix", dict(img=data_root))
        super().__init__(pipeline=pipeline or [], **kwargs)

    def load_data_list(self) -> List[dict]:
        root = self.data_prefix.get("img")
        if not root:
            raise ValueError("data_prefix.img (paired root) is required")
        paired = read_paired(root, validate=True)
        by_cat_cam = {c["id"]: i for i, c in enumerate(
            paired.camera.get("categories", []))}
        by_cat_son = {c["id"]: i for i, c in enumerate(
            paired.sonar.get("categories", []))}

        data_list: List[dict] = []
        for sample in paired.samples():
            if self.split and sample["split"] != self.split:
                continue
            level = self.fixed_deg_level or str(
                sample["meta"].get("deg_level", "d0"))
            # annotations without a source tag (hand labels) always pass
            son_anns = [a for a in sample["sonar_annotations"]
                        if a.get("source", "blob") in self.sonar_gt_sources]
            data_list.append(dict(
                sample_id=sample["sample_id"],
                fp_camera=os.path.join(root, sample["camera_file"])
                if sample["camera_file"] else None,
                fp_sonar=os.path.join(root, sample["sonar_file"])
                if sample["sonar_file"] else None,
                coco_camera_image_id=sample["camera_image_id"],
                coco_sonar_image_id=sample["sonar_image_id"],
                boxes_camera=[a["bbox"] for a in sample["camera_annotations"]],
                labels_camera=[by_cat_cam[a["category_id"]] for a in
                               sample["camera_annotations"]],
                boxes_sonar=[a["bbox"] for a in son_anns],
                labels_sonar=[by_cat_son[a["category_id"]] for a in
                              son_anns],
                deg_level=level,
                deg_seed=int(sample["meta"].get("deg_seed", 0)),
                meta=dict(sample["meta"]),
            ))
        return data_list

    # category metainfo for the sonar side (the camera side keeps its own
    # id space; heads state num_classes explicitly in configs)
    def _get_metainfo(self) -> Dict[str, Any]:
        root = (self.data_prefix or {}).get("img")
        meta: Dict[str, Any] = {}
        if root and os.path.exists(os.path.join(root, "sonar.json")):
            paired = read_paired(root, validate=False)
            cats = paired.sonar.get("categories", [])
            meta["classes"] = tuple(c["name"] for c in cats)
        return meta
