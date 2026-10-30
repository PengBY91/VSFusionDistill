"""End-to-end smoke of tools/train.py -> tools/test.py on synthetic data.

Exercises the real Runner path (train loop, optim wrapper, checkpoint
hook, run_meta stamp) and the inference/eval tool (per-level dataset
override, per-branch prediction dumps, COCO AP) with tiny CPU models —
the same loop W3 runs on the training machine, one scale down.
"""

import json
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")

from tests.test_paired_pipeline import _make_toy_paired

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# tiny two-stream topology (same shape as tests/test_two_stream_model)
_TINY = """
_backbone = dict(type='CSPDarknet', deepen_factor=0.33, widen_factor=0.25,
                 out_indices=(2, 3, 4))
_neck = dict(type='YOLOXPAFPN', in_channels=[64, 128, 256],
             out_channels=64, num_csp_blocks=1)
_head = dict(type='YOLOXHead', num_classes=1, in_channels=64,
             feat_channels=64)

default_scope = 'mmdet'
custom_imports = dict(imports=['aodet.datasets', 'aodet.models'],
                      allow_failed_imports=False)

model = dict(
    type='TwoStreamYOLOX',
    data_preprocessor=dict(type='PairedDataPreprocessor', mean=[0, 0, 0],
                           std=[1, 1, 1], bgr_to_rgb=False,
                           pad_size_divisor=32),
    sonar_backbone=_backbone, sonar_neck=_neck,
    camera_backbone=_backbone, camera_neck=_neck,
    camera_head=_head, uncoupled_sonar_head=_head, fused_sonar_head=_head,
    fusion=dict(type='NoneFusion'),
    train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),
)

train_dataloader = dict(
    batch_size=2, num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(type='PairedSonarCameraDataset', data_root=DATA_ROOT,
                 split='train', sonar_zero_centred=True,
                 pipeline=[
                     dict(type='AOLoadPaired'),
                     dict(type='AOPairedResize', camera_size=(320, 256),
                          sonar_size=(320, 256)),
                     dict(type='AOPairedSyncFlip', prob=0.0),
                     dict(type='AOSampleDegLevel', seed=0),
                     dict(type='AOStudentCameraView', dinf_mode='drop'),
                     dict(type='AOPackPairedInputs'),
                 ]))

test_dataloader = dict(
    batch_size=1, num_workers=0,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(type='PairedSonarCameraDataset', data_root=DATA_ROOT,
                 split='test', sonar_zero_centred=True,
                 pipeline=[
                     dict(type='AOLoadPaired'),
                     dict(type='AOPairedResize', camera_size=(320, 256),
                          sonar_size=(320, 256)),
                     dict(type='AOStudentCameraView', dinf_mode='drop'),
                     dict(type='AOPackPairedInputs'),
                 ]))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=1, val_interval=100)
val_cfg = None
test_cfg = dict(type='TestLoop')
test_evaluator = dict(type='DummyMetric')

optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='SGD', lr=0.01, momentum=0.9))
param_scheduler = [dict(type='ConstantLR', factor=1.0)]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    checkpoint=dict(type='CheckpointHook', interval=1),
)
env_cfg = dict(cudnn_benchmark=False,
               mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)
log_level = 'WARNING'
"""


def _write_config(path: str, data_root: str) -> str:
    cfg = "DATA_ROOT = %r\n" % data_root + _TINY
    with open(path, "w") as f:
        f.write(cfg)
    return path


def test_train_then_test_tools(tmp_path):
    data_root = str(tmp_path / "paired")
    _make_toy_paired_all(data_root)

    cfg = _write_config(str(tmp_path / "tiny.py"), data_root)
    work_dir = str(tmp_path / "work")

    r_train = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "train.py"), cfg,
         "--work-dir", work_dir, "--seed", "0"],
        capture_output=True, text=True, cwd=REPO)
    assert r_train.returncode == 0, r_train.stdout + r_train.stderr
    assert os.path.exists(os.path.join(work_dir, "run_meta.txt"))
    ckpts = [f for f in os.listdir(work_dir) if f.endswith(".pth")]
    assert ckpts, os.listdir(work_dir)

    out_dir = str(tmp_path / "preds")
    r_test = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "test.py"), cfg,
         os.path.join(work_dir, sorted(ckpts)[-1]),
         "--out-dir", out_dir, "--device", "cpu",
         "--levels", "d0", "dinf"],
        capture_output=True, text=True, cwd=REPO)
    assert r_test.returncode == 0, r_test.stdout + r_test.stderr

    for branch in ("camera", "sonar", "sonar_uncoupled"):
        for level in ("d0", "dinf"):
            p = os.path.join(out_dir, f"pred.{branch}.{level}.json")
            assert os.path.exists(p), os.listdir(out_dir)
            with open(p) as f:
                preds = json.load(f)
            for r in preds:
                assert set(r) >= {"image_id", "category_id", "bbox", "score"}
    with open(os.path.join(out_dir, "ap_summary.json")) as f:
        summary = json.load(f)
    assert {(r["branch"], r["level"]) for r in summary} == {
        (b, l) for b in ("camera", "sonar", "sonar_uncoupled")
        for l in ("d0", "dinf")}
    for r in summary:
        assert 0.0 <= r["AP"] <= 1.0


def _make_toy_paired_all(root: str):
    """One paired root containing both train and test splits."""
    import numpy as np
    from aodet.datasets.paired_format import PairedData, write_paired
    import cv2

    rng = np.random.default_rng(7)
    for sub in ("camera", "sonar"):
        os.makedirs(os.path.join(root, "images", sub), exist_ok=True)
    cam_imgs, son_imgs, cam_anns, son_anns = [], [], [], []
    pairs, meta = [], []
    ann_id = 0
    for split, n in (("train", 4), ("test", 2)):
        prefix = "tr" if split == "train" else "te"
        for i in range(n):
            cfp = f"images/camera/{prefix}c{i}.jpg"
            sfp = f"images/sonar/{prefix}s{i}.png"
            img = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(root, cfp), img)
            cv2.imwrite(os.path.join(root, sfp), img)
            cid, sid = len(cam_imgs) + 1, len(son_imgs) + 1
            cam_imgs.append(dict(id=cid, file_name=cfp, width=320,
                                 height=240))
            son_imgs.append(dict(id=sid, file_name=sfp, width=320,
                                 height=240))
            cam_anns.append(dict(id=ann_id, image_id=cid, category_id=1,
                                 bbox=[50, 60, 40, 30], area=1200,
                                 iscrowd=0))
            son_anns.append(dict(id=ann_id, image_id=sid, category_id=1,
                                 bbox=[140, 120, 30, 20], area=600,
                                 iscrowd=0))
            pairs.append(dict(camera_ann_id=ann_id, sonar_ann_id=ann_id,
                              sample_id=f"{prefix}{i}"))
            meta.append(dict(sample_id=f"{prefix}{i}", camera_image_id=cid,
                             sonar_image_id=sid, split=split,
                             deg_level="d0", deg_seed="0"))
            ann_id += 1
    write_paired(root, PairedData(
        camera=dict(images=cam_imgs, annotations=cam_anns,
                    categories=[dict(id=1, name="uxo")]),
        sonar=dict(images=son_imgs, annotations=son_anns,
                   categories=[dict(id=1, name="uxo")]),
        pairs=pairs, frame_meta=meta))
