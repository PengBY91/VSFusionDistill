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


def test_sonar_scale_jitter_rows_and_boxes():
    """E08: jitter rescales the sonar range axis and box y/h only, on a
    fixed canvas; camera stream and azimuth (x/w) stay fixed."""
    from aodet.datasets.transforms import AOSonarScaleJitter
    img_sonar = np.zeros((100, 50, 3), np.uint8)
    img_sonar[40:52, 20:30] = 255
    img_camera = np.zeros((30, 40, 3), np.uint8)
    t = AOSonarScaleJitter(scale_range=(1.5, 1.5))
    out = t.transform(dict(img_sonar=img_sonar.copy(),
                           img_camera=img_camera.copy(),
                           boxes_sonar=[[20.0, 40.0, 10.0, 12.0]],
                           labels_sonar=[2],
                           boxes_camera=[[1.0, 2.0, 3.0, 4.0]]))
    assert out["img_sonar"].shape == (100, 50, 3)   # canvas preserved
    assert out["img_camera"].shape == (30, 40, 3)
    (x, y, w, h), = out["boxes_sonar"]
    assert (x, w) == (20.0, 10.0)                   # azimuth untouched
    assert h == pytest.approx(18.0)                 # 12 * 1.5
    assert 0.0 <= y and y + h <= 100.0              # box kept in window
    rows = np.where(out["img_sonar"][:, 25, 0] > 127)[0]
    assert rows.min() == pytest.approx(y, abs=1.5)
    assert rows.max() + 1 == pytest.approx(y + h, abs=1.5)
    assert out["labels_sonar"] == [2]
    assert out["boxes_camera"] == [[1.0, 2.0, 3.0, 4.0]]
    assert out["sonar_scale_jitter"] == 1.5


def test_sonar_scale_jitter_shrink_pads():
    from aodet.datasets.transforms import AOSonarScaleJitter
    img = np.full((100, 50, 3), 9, np.uint8)
    out = AOSonarScaleJitter(scale_range=(0.5, 0.5), pad_val=0).transform(
        dict(img_sonar=img.copy(), boxes_sonar=[[1.0, 20.0, 3.0, 40.0]],
             labels_sonar=[0]))
    assert out["img_sonar"].shape == (100, 50, 3)
    assert (out["img_sonar"][:, 0, 0] == 9).sum() == 50
    assert out["boxes_sonar"][0][3] == pytest.approx(20.0)


def test_scale_jitter_survives_full_train_pipeline():
    """Review 2026-10-31 §1.2 regression: placed after the fixed-size
    resize, the jitter must change the final box height (the first E08
    config put it before the resize, which undid it exactly)."""
    from mmcv.transforms import Compose as _Compose
    img = np.zeros((441, 128, 3), np.uint8)
    img[200:217, 60:70] = 255
    heights = []
    for f in (0.6, 1.8):
        pipe = _Compose([
            dict(type="AOPairedResize", sonar_size=(640, 640),
                 keep_ratio=False),
            dict(type="AOSonarScaleJitter", scale_range=(f, f)),
        ])
        out = pipe(dict(img_sonar=img.copy(), img_camera=None,
                        boxes_sonar=[[60.0, 200.0, 10.0, 17.0]],
                        labels_sonar=[0], boxes_camera=[]))
        assert out["img_sonar"].shape[:2] == (640, 640)
        heights.append(out["boxes_sonar"][0][3])
    assert heights[1] / heights[0] == pytest.approx(1.8 / 0.6, rel=0.02)


def test_e08_configs_jitter_after_resize():
    from mmengine.config import Config
    jittered = ("configs/f2/e08_s0_b0_sonar_only_yoloxs.py",
                "configs/f2/e08_s0_cross_attn_yoloxs.py",
                "configs/f2/e09_s0_b0_jitter_yoloxs.py",
                "configs/f2/e09_s0_cross_attn_yoloxs.py")
    for path in jittered:
        types = [t["type"] for t in
                 Config.fromfile(path).train_dataloader.dataset.pipeline]
        assert types.index("AOSonarScaleJitter") > \
            types.index("AOPairedResize"), (path, types)
    # E09a is the recipe-only control: it must carry NO jitter, or the
    # a/b contrast (recipe effect vs jitter effect) collapses
    types = [t["type"] for t in Config.fromfile(
        "configs/f2/e09_s0_b0_recipe_yoloxs.py"
    ).train_dataloader.dataset.pipeline]
    assert "AOSonarScaleJitter" not in types


def test_dinf_clears_camera_gt_and_flags_sample():
    """Review §1.5: a D∞ frame carries no camera target (black frame or
    dropped); its camera GT is cleared and camera_valid is False."""
    from aodet.datasets.transforms import AOStudentCameraView
    for mode in ("zero", "drop"):
        out = AOStudentCameraView(dinf_mode=mode).transform(dict(
            img_camera=np.full((8, 8, 3), 128, np.uint8),
            boxes_camera=[[1, 1, 3, 3]], labels_camera=[0],
            deg_level="dinf", sample_id="x"))
        assert out["boxes_camera"] == [] and out["labels_camera"] == []
        assert out["camera_valid"] is False
        assert out["img_camera_clear"].max() == 128  # teacher still sees
        if mode == "zero":
            assert out["img_camera"].max() == 0
        else:
            assert out["img_camera"] is None
    out = AOStudentCameraView().transform(dict(
        img_camera=np.full((8, 8, 3), 128, np.uint8),
        boxes_camera=[[1, 1, 3, 3]], labels_camera=[0],
        deg_level="d2", sample_id="x"))
    assert out["camera_valid"] is True and len(out["boxes_camera"]) == 1


def test_sample_deg_level_lazy_rng_follows_global_seed():
    """Review §1.7: without an explicit seed, the level/seed draws derive
    from the (worker-seeded) global numpy RNG, not a fixed seed 0."""
    from aodet.datasets.transforms import AOSampleDegLevel

    def draws(global_seed):
        np.random.seed(global_seed)
        t = AOSampleDegLevel()
        return [(r["deg_level"], r["deg_seed"]) for r in
                (t.transform({}) for _ in range(20))]

    assert draws(1) == draws(1)
    assert draws(1) != draws(2)
    fixed = AOSampleDegLevel(seed=0, resample_deg_seed=False)
    assert fixed.transform({"deg_seed": 7})["deg_seed"] == 7


def test_sonar_scale_jitter_identity_paths():
    from aodet.datasets.transforms import AOSonarScaleJitter
    img = np.zeros((100, 50, 3), np.uint8)
    for kwargs in (dict(scale_range=(1, 1)), dict(prob=0.0)):
        t = AOSonarScaleJitter(**kwargs)
        out = t.transform(dict(img_sonar=img.copy(),
                               boxes_sonar=[[1.0, 2.0, 3.0, 4.0]]))
        assert out["img_sonar"].shape == (100, 50, 3)
        assert out["boxes_sonar"] == [[1.0, 2.0, 3.0, 4.0]]
        assert out["sonar_scale_jitter"] == 1.0
    out = AOSonarScaleJitter().transform(dict(img_sonar=None))
    assert out["sonar_scale_jitter"] == 1.0


def test_sonar_scale_jitter_rejects_bad_range():
    from aodet.datasets.transforms import AOSonarScaleJitter
    with pytest.raises(ValueError):
        AOSonarScaleJitter(scale_range=(1.8, 0.6))
