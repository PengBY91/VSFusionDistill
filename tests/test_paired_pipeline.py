"""End-to-end pipeline test: paired format on disk -> transforms ->
preprocessor -> TwoStreamYOLOX loss.  Uses synthetic images (no real data
needed), exercising exactly the training path of S0-lite.
"""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mmdet")

import aodet.models  # noqa: F401
import aodet.datasets.transforms  # noqa: F401
from mmengine.registry import init_default_scope

init_default_scope("mmdet")

from mmcv.transforms import Compose

from aodet.datasets import PairedSonarCameraDataset
from aodet.datasets.paired_format import PairedData, write_paired
from aodet.models.detectors.two_stream_yolox import TwoStreamYOLOX
from tests.test_two_stream_model import _stream_cfg

PIPE = dict(
    type="Compose",
    transforms=[
        dict(type="AOLoadPaired"),
        dict(type="AOPairedResize", camera_size=(320, 256),
             sonar_size=(320, 256)),
        dict(type="AOPairedSyncFlip", prob=0.0),
        dict(type="AOSampleDegLevel", seed=0),
        dict(type="AOStudentCameraView", dinf_mode="drop"),
        dict(type="AOPackPairedInputs"),
    ],
)


def _make_toy_paired(root: str, n: int = 4, split: str = "train"):
    rng = np.random.default_rng(7)
    os.makedirs(os.path.join(root, "images", "camera"), exist_ok=True)
    os.makedirs(os.path.join(root, "images", "sonar"), exist_ok=True)
    cam_imgs, son_imgs, cam_anns, son_anns, pairs, meta = [], [], [], [], [], []
    ann_id = 0
    for i in range(n):
        cfp = f"images/camera/c{i:03d}.jpg"
        sfp = f"images/sonar/s{i:03d}.png"
        cv2_img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
        import cv2
        cv2.imwrite(os.path.join(root, cfp), cv2_img)
        cv2.imwrite(os.path.join(root, sfp), cv2_img)
        cid, sid = i + 1, i + 1
        cam_imgs.append(dict(id=cid, file_name=cfp, width=320, height=240))
        son_imgs.append(dict(id=sid, file_name=sfp, width=320, height=240))
        cam_anns.append(dict(id=ann_id, image_id=cid, category_id=1,
                             bbox=[50, 60, 40, 30], area=1200, iscrowd=0))
        son_anns.append(dict(id=ann_id, image_id=sid, category_id=1,
                             bbox=[140, 120, 30, 20], area=600, iscrowd=0))
        pairs.append(dict(camera_ann_id=ann_id, sonar_ann_id=ann_id,
                          sample_id=f"s{i:03d}"))
        meta.append(dict(sample_id=f"s{i:03d}", camera_image_id=cid,
                         sonar_image_id=sid, split=split,
                         deg_level="d0", deg_seed="0"))
        ann_id += 1
    write_paired(root, PairedData(
        camera=dict(images=cam_imgs, annotations=cam_anns,
                    categories=[dict(id=1, name="uxo")]),
        sonar=dict(images=son_imgs, annotations=son_anns,
                   categories=[dict(id=1, name="uxo")]),
        pairs=pairs, frame_meta=meta))


def test_full_pipeline_and_loss(tmp_path):
    root = str(tmp_path / "paired")
    _make_toy_paired(root)

    ds = PairedSonarCameraDataset(
        data_root=root, split="train",
        pipeline=PIPE["transforms"],
        serialize_data=False, lazy_init=False)
    assert len(ds) == 4

    item = ds[0]
    for key in ("img_sonar", "img_camera", "img_camera_clear",
                "data_samples_sonar", "data_samples_camera", "deg_level"):
        assert key in item, (key, list(item))
    # deg draw consumed: camera differs from the clean fork
    if item["deg_level"] != "d0":
        assert not np.array_equal(item["img_camera"],
                                  item["img_camera_clear"])
    assert item["img_sonar"].shape == (3, 256, 320)

    # batch of 2 through the model
    from mmengine.dataset import pseudo_collate
    batch = pseudo_collate([ds[0], ds[1]])
    sbb, snc, shd = _stream_cfg()
    model = TwoStreamYOLOX(
        sonar_backbone=sbb, sonar_neck=snc,
        camera_backbone=_stream_cfg()[0], camera_neck=_stream_cfg()[1],
        camera_head=_stream_cfg()[2],
        fused_sonar_head=shd,
        uncoupled_sonar_head=_stream_cfg()[2],
        fusion=dict(type="ConcatFusion", cam_channels=[64, 64, 64],
                    sonar_channels=[64, 64, 64], out_channels=64),
        train_cfg=dict(assigner=dict(type="SimOTAAssigner", center_radius=2.5)),
        test_cfg=dict(score_thr=0.01,
                      nms=dict(type="nms", iou_threshold=0.65)),
        data_preprocessor=dict(type="PairedDataPreprocessor",
                               mean=[0, 0, 0], std=[1, 1, 1],
                               bgr_to_rgb=False, pad_size_divisor=32),
    )
    prepped = model.data_preprocessor(batch, training=True)
    model.train()
    losses = model.loss(prepped["inputs"], prepped["data_samples"])
    assert any(k.startswith("camera/") for k in losses)
    assert any(k.startswith("fused_sonar/") for k in losses)
    assert any(k.startswith("uncoupled_sonar/") for k in losses)
    total = sum(v for k, v in losses.items() if "loss" in k)
    total.backward()


def test_dinf_drop_path(tmp_path):
    root = str(tmp_path / "paired")
    _make_toy_paired(root)
    ds = PairedSonarCameraDataset(
        data_root=root, split="train",
        pipeline=[
            dict(type="AOLoadPaired"),
            dict(type="AOPairedResize", camera_size=(320, 256),
                 sonar_size=(320, 256)),
            dict(type="AOPairedSyncFlip", prob=0.0),
            # force the D-inf endpoint
            dict(type="AOSampleDegLevel",
                 probs=dict(d0=0, d1=0, d2=0, d3=0, d4=0, dinf=1.0)),
            dict(type="AOStudentCameraView", dinf_mode="drop"),
            dict(type="AOPackPairedInputs"),
        ],
        serialize_data=False, lazy_init=False)
    item = ds[0]
    assert item["deg_level"] == "dinf"
    assert item["img_camera"] is None          # dropped, not zeroed
    assert item["img_camera_clear"] is not None  # teacher still sees clear
    # sonar side untouched
    assert item["img_sonar"] is not None
    assert item["data_samples_camera"] is None


def test_sonar_gt_sources_filter(tmp_path):
    """E05: sonar_gt_sources=("blob",) drops projected sonar boxes from
    training targets; camera annotations are untouched."""
    root = str(tmp_path / "paired")
    _make_toy_paired(root, n=4)
    import json as _json
    p = os.path.join(root, "sonar.json")
    with open(p) as f:
        sonar = _json.load(f)
    # half the frames keep measured echoes, half become projected
    for i, a in enumerate(sonar["annotations"]):
        a["source"] = "blob" if i % 2 == 0 else "projected"
    with open(p, "w") as f:
        _json.dump(sonar, f)

    ds = PairedSonarCameraDataset(
        data_root=root, split="train", sonar_gt_sources=("blob",),
        pipeline=[])
    assert all(len(d["boxes_sonar"]) == (1 if i % 2 == 0 else 0)
               for i, d in enumerate(ds.load_data_list()))
    assert all(len(d["boxes_camera"]) == 1
               for d in ds.load_data_list())
    ds_all = PairedSonarCameraDataset(
        data_root=root, split="train", pipeline=[])
    assert all(len(d["boxes_sonar"]) == 1 for d in ds_all.load_data_list())
