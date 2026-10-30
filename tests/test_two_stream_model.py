"""Smoke tests for TwoStreamYOLOX on CPU with real mmdet components.

Covers: three-head loss + backward, per-branch predict, B0 sonar-only
configuration, the online frozen-teacher KD path, and the cross-plane MIL
loss.  Sizes are nano-scale so the whole file runs in seconds on CPU.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("mmdet")
mmcv = pytest.importorskip("mmcv")

import aodet.models  # noqa: F401  (registers the extensions)
from mmengine.registry import init_default_scope

init_default_scope("mmdet")  # configs set default_scope='mmdet' too
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData
from mmdet.structures.bbox import HorizontalBoxes

from aodet.geometry.m0 import M0Mapping
from aodet.models.detectors.two_stream_yolox import TwoStreamYOLOX
from aodet.models.losses.cross_plane_mil import CrossPlaneMILLoss

IMG = (256, 320)  # H, W — divisible by 32


def _stream_cfg(num_classes=3):
    norm = dict(type="BN", momentum=0.03, eps=0.001)
    act = dict(type="Swish")
    common = dict(norm_cfg=norm, act_cfg=act)
    backbone = dict(type="CSPDarknet", deepen_factor=0.33, widen_factor=0.25,
                    out_indices=(2, 3, 4), **common)
    neck = dict(type="YOLOXPAFPN", in_channels=[64, 128, 256],
                out_channels=64, num_csp_blocks=1, **common)
    head = dict(type="YOLOXHead", num_classes=num_classes, in_channels=64,
                feat_channels=64, stacked_convs=2, strides=(8, 16, 32),
                **common,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=True,
                              reduction="sum", loss_weight=1.0),
                loss_bbox=dict(type="IoULoss", mode="square", eps=1e-16,
                               reduction="sum", loss_weight=5.0),
                loss_obj=dict(type="CrossEntropyLoss", use_sigmoid=True,
                              reduction="sum", loss_weight=1.0),
                loss_l1=dict(type="L1Loss", reduction="sum",
                             loss_weight=1.0))
    return backbone, neck, head


def _model(fusion=None, with_camera=True, distillation=None):
    sbb, snc, shd = _stream_cfg()
    kwargs = dict(
        sonar_backbone=sbb, sonar_neck=snc,
        uncoupled_sonar_head=shd,
        train_cfg=dict(assigner=dict(type="SimOTAAssigner",
                                     center_radius=2.5)),
        test_cfg=dict(score_thr=0.01,
                      nms=dict(type="nms", iou_threshold=0.65)),
        data_preprocessor=dict(type="PairedDataPreprocessor",
                               mean=[0, 0, 0], std=[1, 1, 1],
                               bgr_to_rgb=False, pad_size_divisor=32),
    )
    if with_camera:
        cbb, cnc, chd = _stream_cfg()
        kwargs.update(camera_backbone=cbb, camera_neck=cnc,
                      camera_head=chd)
        if fusion is not None:
            kwargs.update(fused_sonar_head=_stream_cfg()[2],
                          fusion=fusion)
    if distillation:
        kwargs["distillation"] = distillation
    return TwoStreamYOLOX(**kwargs)


def _sample(n_obj=2, prefix="img"):
    ds = DetDataSample()
    ds.set_metainfo({
        "img_id": 0,
        "img_shape": IMG,
        "ori_shape": IMG,
        "scale_factor": (1.0, 1.0),
        "batch_input_shape": IMG,
        "pad_shape": IMG,
    })
    gt = InstanceData()
    gt.bboxes = HorizontalBoxes(
        torch.tensor([[40.0, 40.0, 90.0, 80.0],
                      [150.0, 120.0, 220.0, 190.0]][:n_obj],
                     dtype=torch.float32))
    gt.labels = torch.zeros(n_obj, dtype=torch.long)
    ds.gt_instances = gt
    return ds


def _batch(b=2, with_clear=False, n_obj=2):
    import numpy as np
    rng = np.random.default_rng(0)
    data = {
        "inputs": None,
        "img_sonar": [torch.from_numpy(
            rng.integers(0, 255, (3, *IMG), dtype=np.uint8))
            for _ in range(b)],
        "img_camera": [torch.from_numpy(
            rng.integers(0, 255, (3, *IMG), dtype=np.uint8))
            for _ in range(b)],
        "data_samples_sonar": [_sample(n_obj) for _ in range(b)],
        "data_samples_camera": [_sample(n_obj) for _ in range(b)],
    }
    if with_clear:
        data["img_camera_clear"] = list(data["img_camera"])
    return data


def _prep(model, data, training):
    prepped = model.data_preprocessor(data, training=training)
    return prepped["inputs"], prepped["data_samples"]


def test_fused_model_loss_and_backward():
    model = _model(fusion=dict(type="CrossAttnFusion", cam_channels=[64, 64, 64],
                               sonar_channels=[64, 64, 64], embed_dim=64,
                               num_heads=4))
    assert model.camera_head is not None
    assert model.fused_sonar_head is not None
    assert model.uncoupled_sonar_head is not None
    data = _batch(b=2)
    inputs, samples = _prep(model, data, training=True)
    model.train()
    losses = model.loss(inputs, samples)
    for prefix in ("camera/", "fused_sonar/", "uncoupled_sonar/"):
        assert any(k.startswith(prefix) for k in losses), (prefix, losses)
    total = sum(v for k, v in losses.items() if "loss" in k)
    total.backward()


def test_fused_model_predict_branches():
    model = _model(fusion=dict(type="ConcatFusion", cam_channels=[64, 64, 64],
                               sonar_channels=[64, 64, 64], out_channels=64))
    model.eval()
    data = _batch(b=1)
    inputs, samples = _prep(model, data, training=False)
    with torch.no_grad():
        out = model.predict(inputs, samples)
    assert set(out) == {"camera", "sonar", "sonar_uncoupled"}
    for branch, samples in out.items():
        assert len(samples) == 1
        pi = samples[0].pred_instances
        # attached object is InstanceData, not a nested DetDataSample
        assert not hasattr(pi, "pred_instances")
        for attr in ("bboxes", "scores", "labels"):
            assert hasattr(pi, attr)


def test_predict_camera_branch_at_dinf_drop():
    """D∞ drop: camera input None -> camera branch emits empty predictions."""
    model = _model(fusion=dict(type="NoneFusion"))
    model.eval()
    data = _batch(b=2)
    del data["img_camera"]
    inputs, samples = _prep(model, data, training=False)
    assert inputs["camera"] is None
    with torch.no_grad():
        out = model.predict(inputs, samples)
    assert len(out["camera"]) == 2
    for ds in out["camera"]:
        assert len(ds.pred_instances) == 0


def test_b0_sonar_only():
    model = _model(with_camera=False)
    assert model.camera_backbone is None
    data = _batch(b=2)
    del data["img_camera"], data["data_samples_camera"]
    inputs, samples = _prep(model, data, training=True)
    assert inputs["camera"] is None
    losses = model.loss(inputs, samples)
    assert not any(k.startswith("camera/") for k in losses)
    assert any(k.startswith("uncoupled_sonar/") for k in losses)
    sum(v for k, v in losses.items() if "loss" in k).backward()


def test_kd_path_with_frozen_teacher():
    sbb, snc, shd = _stream_cfg()
    teacher_cfg = dict(
        type="TwoStreamYOLOX",
        sonar_backbone=sbb, sonar_neck=snc,
        camera_backbone=_stream_cfg()[0], camera_neck=_stream_cfg()[1],
        camera_head=_stream_cfg()[2], uncoupled_sonar_head=shd)
    student = _model(
        fusion=dict(type="NoneFusion"),
        distillation=dict(teacher=teacher_cfg, teacher_input="clear",
                          obj_weight=1.0, cls_weight=0.0))
    data = _batch(b=2, with_clear=True)
    inputs, samples = _prep(student, data, training=True)
    student.train()
    losses = student.loss(inputs, samples)
    kd_keys = [k for k in losses if k.startswith("kd/")]
    assert any("obj" in k for k in kd_keys), kd_keys
    assert all(not p.requires_grad for p in student.teacher.parameters())
    sum(v for k, v in losses.items() if "loss" in k).backward()


def test_cross_plane_mil_loss():
    m0 = M0Mapping(slope=30.75, intercept=-15.375, residual_std=0.5,
                   camera_hfov_deg=30.75, sonar_fov_deg=30.75)
    crit = CrossPlaneMILLoss(m0=m0, bearing_fov_deg=30.75)
    # teacher sees a box in the camera centre -> window around bearing 0
    teacher_boxes = [[(140.0, 100.0, 180.0, 140.0)]]
    teacher_obj = [torch.tensor([0.9])]
    # sonar objectness map with a bump at the centre column
    obj = torch.full((1, 1, 8, 10), -4.0)
    obj[0, 0, 4, 5] = 3.0
    loss, n_used = crit(teacher_boxes, teacher_obj, [obj],
                        camera_img_w=320)
    assert n_used.item() == 1
    assert loss.item() >= 0
    # empty batch -> zero loss, zero used
    loss0, n0 = crit([[]], [torch.tensor([])], [obj], camera_img_w=320)
    assert n0.item() == 0 and loss0.item() == 0.0
