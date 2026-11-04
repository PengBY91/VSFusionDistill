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


def test_predict_sonar_branches_do_not_alias():
    """E07 regression: mmdet's add_pred_to_datasample mutates samples in
    place; the fused and uncoupled sonar heads share one sample list, so
    the second branch used to overwrite the first and both prediction
    files ended up bit-identical.  Each branch must own fresh samples and
    genuinely different predictions when the heads differ."""
    model = _model(fusion=dict(type="ConcatFusion", cam_channels=[64, 64, 64],
                               sonar_channels=[64, 64, 64], out_channels=64))
    with torch.no_grad():      # make the two heads provably different
        for p in model.uncoupled_sonar_head.parameters():
            p.add_(0.05)
    model.eval()
    data = _batch(b=2)
    inputs, samples = _prep(model, data, training=False)
    with torch.no_grad():
        out = model.predict(inputs, samples)
    a, b = out["sonar"], out["sonar_uncoupled"]
    assert a[0] is not b[0] and a[1] is not b[1]    # no shared objects
    assert all(x is not y for x, y in zip(a, b))
    # the original input samples stay untouched (no in-place writes)
    orig = samples["sonar"]
    assert all(not hasattr(ds, "pred_instances") for ds in orig)
    # different heads -> at least one frame's detections differ (unequal
    # counts count: random-init heads keep different numbers above thr)
    differs = any(
        len(sa.pred_instances) != len(sb.pred_instances)
        or not torch.allclose(sa.pred_instances.scores,
                              sb.pred_instances.scores)
        or not torch.allclose(sa.pred_instances.bboxes,
                              sb.pred_instances.bboxes)
        for sa, sb in zip(a, b))
    assert differs


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


def test_bcam_camera_only():
    """E11 Bcam (B-class amendment 4): both sonar heads None -- the
    sonar forward is skipped and only the camera branch appears in
    loss and predict."""
    sbb, snc, _ = _stream_cfg()
    cbb, cnc, chd = _stream_cfg()
    model = TwoStreamYOLOX(
        sonar_backbone=sbb, sonar_neck=snc,
        camera_backbone=cbb, camera_neck=cnc, camera_head=chd,
        uncoupled_sonar_head=None, fused_sonar_head=None,
        train_cfg=dict(assigner=dict(type="SimOTAAssigner",
                                     center_radius=2.5)),
        test_cfg=dict(score_thr=0.01,
                      nms=dict(type="nms", iou_threshold=0.65)),
        data_preprocessor=dict(type="PairedDataPreprocessor",
                               mean=[0, 0, 0], std=[1, 1, 1],
                               bgr_to_rgb=False, pad_size_divisor=32),
    )
    assert model.fused_sonar_head is None and \
        model.uncoupled_sonar_head is None
    calls = {"n": 0}
    orig = model.extract_sonar_feat

    def _spy(img):
        calls["n"] += 1
        return orig(img)

    model.extract_sonar_feat = _spy
    data = _batch(b=2)
    inputs, samples = _prep(model, data, training=True)
    losses = model.loss(inputs, samples)
    assert losses and all(k.startswith("camera/") for k in losses)
    sum(v for k, v in losses.items() if "loss" in k).backward()
    inputs, samples = _prep(model, _batch(b=1), training=False)
    out = model.predict(inputs, samples)
    assert set(out) == {"camera"}
    assert calls["n"] == 0


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
                          allow_random_teacher=True,
                          obj_weight=1.0, cls_weight=0.0))
    data = _batch(b=2, with_clear=True)
    inputs, samples = _prep(student, data, training=True)
    student.train()
    losses = student.loss(inputs, samples)
    kd_keys = [k for k in losses if k.startswith("kd/")]
    assert any("obj" in k for k in kd_keys), kd_keys
    assert all(not p.requires_grad for p in student.teacher.parameters())
    sum(v for k, v in losses.items() if "loss" in k).backward()


def test_frozen_teacher_loads_real_checkpoint(tmp_path):
    """torch>=2.6 defaults torch.load to weights_only=True; mmengine
    checkpoints embed non-tensor meta objects and the strict unpickler
    rejects them.  FrozenTeacher must load them anyway (E10: the teacher
    checkpoint is this repo's own artifact)."""
    import numpy as np
    from aodet.models.teachers.frozen_teacher import FrozenTeacher
    sbb, snc, shd = _stream_cfg()
    teacher_cfg = dict(
        type="TwoStreamYOLOX",
        sonar_backbone=sbb, sonar_neck=snc,
        camera_backbone=_stream_cfg()[0], camera_neck=_stream_cfg()[1],
        camera_head=_stream_cfg()[2], uncoupled_sonar_head=shd)
    reference = FrozenTeacher(dict(teacher=teacher_cfg,
                                   allow_random_teacher=True)).model
    # a checkpoint the way mmengine saves it: state_dict + non-tensor meta
    path = str(tmp_path / "teacher.pth")
    torch.save({"state_dict": reference.state_dict(),
                "meta": {"seed": np.int64(0), "iter": np.array([1, 2])}},
               path)
    # premise only where it holds: torch>=2.6 defaults weights_only=True
    import inspect
    strict_default = bool(
        inspect.signature(torch.load).parameters["weights_only"].default)
    if strict_default:
        with pytest.raises(Exception):
            torch.load(path, map_location="cpu")
    loaded = FrozenTeacher(dict(teacher=teacher_cfg, teacher_checkpoint=path))
    for (ka, va), (kb, vb) in zip(reference.state_dict().items(),
                                  loaded.model.state_dict().items()):
        assert ka == kb and torch.equal(va, vb)


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


# -- review 2026-10-31 regressions ----------------------------------------

def _kd_student(**kd):
    sbb, snc, shd = _stream_cfg()
    teacher_cfg = dict(
        type="TwoStreamYOLOX",
        sonar_backbone=sbb, sonar_neck=snc,
        camera_backbone=_stream_cfg()[0], camera_neck=_stream_cfg()[1],
        camera_head=_stream_cfg()[2], uncoupled_sonar_head=shd)
    cfg = dict(teacher=teacher_cfg, teacher_input="clear",
               allow_random_teacher=True,
               obj_weight=1.0, score_weight=0.0)
    cfg.update(kd)
    return _model(fusion=dict(type="CrossAttnFusion",
                              cam_channels=[64, 64, 64],
                              sonar_channels=[64, 64, 64], embed_dim=64,
                              num_heads=4),
                  distillation=cfg)


def test_kd_loss_reaches_student_gradients():
    """§1.1: KD terms must survive mmengine's parse_losses (keys need
    "loss") and produce gradients in the student."""
    student = _kd_student(score_weight=0.5)
    data = _batch(b=2, with_clear=True)
    inputs, samples = _prep(student, data, training=True)
    student.train()
    losses = student.loss(inputs, samples)
    _, log_vars = student.parse_losses(losses)
    kd_keys = [k for k in log_vars if k.startswith("kd/")]
    assert kd_keys and all("loss" in k for k in kd_keys), kd_keys
    # routes: camera, fused sonar and the uncoupled sonar head (§3.3)
    for b in ("camera", "sonar", "sonar_uncoupled"):
        assert f"kd/loss_{b}_obj" in losses, list(losses)
    student.zero_grad()
    sum(v for k, v in losses.items() if k.startswith("kd/")).backward()
    g = sum(float(p.grad.abs().sum())
            for n, p in student.named_parameters()
            if p.requires_grad and p.grad is not None
            and not n.startswith("teacher."))
    assert g > 0
    assert all(p.grad is None for p in student.teacher.parameters())


def test_feature_kd_and_reach():
    student = _kd_student(feat_weight=1.0, student_feat_channels=[64] * 3,
                          teacher_feat_channels=[64] * 3,
                          reach=dict(k=10.0, theta=0.3))
    data = _batch(b=2, with_clear=True)
    inputs, samples = _prep(student, data, training=True)
    student.train()
    losses = student.loss(inputs, samples)
    assert "kd/loss_sonar_feat" in losses
    losses["kd/loss_sonar_feat"].backward()
    assert student.kd_adapters[0].weight.grad is not None


def test_detection_losses_unchanged_by_single_forward():
    """§1.6: the single-forward loss() matches head.loss() term by term."""
    torch.manual_seed(0)
    model = _model(fusion=dict(type="NoneFusion"))
    model.train()
    data = _batch(b=2)
    inputs, samples = _prep(model, data, training=True)
    losses = model.loss(inputs, samples)
    son = model.extract_sonar_feat(inputs["sonar"])
    ref = model.uncoupled_sonar_head.loss(son, samples["sonar"])
    for k, v in ref.items():
        torch.testing.assert_close(
            losses[f"uncoupled_sonar/{k}"],
            v * model.loss_weights["uncoupled_sonar"])


@pytest.mark.parametrize("fusion", [
    dict(type="CrossAttnFusion", cam_channels=[16], sonar_channels=[16],
         embed_dim=16, num_heads=2, layer_scale_init=0.5),
    dict(type="BearingAxialFusion", cam_channels=[16], sonar_channels=[16],
         embed_dim=16, num_heads=2, layer_scale_init=0.5),
    dict(type="ConcatFusion", cam_channels=[16], sonar_channels=[16],
         out_channels=16),
])
def test_fusion_dinf_train_test_consistent(fusion):
    """§2.1: a masked sample (camera_valid False, training with black
    frames) and a dropped camera (test) give the same fused features."""
    from aodet.models.fusion import build_fusion
    f = build_fusion(fusion).eval()
    son = [torch.randn(2, 16, 6, 5)]
    cam = [torch.randn(2, 16, 8, 7)]   # unregistered, different size
    masked = f(cam, son, camera_valid=torch.tensor([False, False]))
    dropped = f(None, son)
    torch.testing.assert_close(masked[0], dropped[0])
    mixed = f(cam, son, camera_valid=torch.tensor([True, False]))
    torch.testing.assert_close(mixed[0][1], dropped[0][1])
    if fusion["type"] != "ConcatFusion":
        assert not torch.allclose(mixed[0][0], dropped[0][0])


def test_cross_attn_zero_layer_scale_is_sonar_only():
    from aodet.models.fusion import CrossAttnFusion
    f = CrossAttnFusion([16], [16], embed_dim=16, num_heads=2)
    son, cam = [torch.randn(1, 16, 4, 4)], [torch.randn(1, 16, 4, 4)]
    torch.testing.assert_close(f(cam, son)[0], f(None, son)[0])


def test_cross_attn_is_position_aware():
    """§2.2: with positional encodings, permuting camera tokens changes
    the fused output (a pure content attention would be invariant)."""
    from aodet.models.fusion import CrossAttnFusion
    torch.manual_seed(0)
    f = CrossAttnFusion([16], [16], embed_dim=16, num_heads=2,
                        layer_scale_init=1.0).eval()
    son = [torch.randn(1, 16, 4, 4)]
    cam = torch.randn(1, 16, 4, 4)
    perm = cam.flatten(2)[:, :, torch.randperm(16)].reshape(1, 16, 4, 4)
    assert not torch.allclose(f([cam], son)[0], f([perm], son)[0])


def test_bearing_attn_geometry_bias_shapes():
    from aodet.models.fusion import BearingAxialFusion
    f = BearingAxialFusion([16], [16], embed_dim=16, num_heads=2,
                           layer_scale_init=1.0,
                           sonar_col_bearing=lambda x: (x - 0.5) * 30.0,
                           camera_col_bearing=lambda x: (x - 0.5) * 90.0)
    out = f([torch.randn(2, 16, 8, 12)], [torch.randn(2, 16, 6, 5)])
    assert out[0].shape == (2, 16, 6, 5)
    out[0].sum().backward()
    assert f.log_sigma.grad is not None


def test_frozen_teacher_rejects_mismatched_checkpoint(tmp_path):
    """§3.5: a key mismatch must raise instead of leaving random weights."""
    from aodet.models.teachers.frozen_teacher import FrozenTeacher
    sbb, snc, shd = _stream_cfg()
    tcfg = dict(type="TwoStreamYOLOX", sonar_backbone=sbb, sonar_neck=snc,
                uncoupled_sonar_head=shd)
    src = TwoStreamYOLOX(**{k: v for k, v in tcfg.items() if k != "type"})
    good = tmp_path / "good.pth"
    torch.save({"state_dict": src.state_dict()}, good)
    t = FrozenTeacher(dict(teacher=tcfg, teacher_checkpoint=str(good)))
    for (n, a), b in zip(src.state_dict().items(),
                         t.model.state_dict().values()):
        torch.testing.assert_close(a, b)
    bad = tmp_path / "bad.pth"
    torch.save({"state_dict": {f"model.{k}": v for k, v in
                               src.state_dict().items()}}, bad)
    with pytest.raises(RuntimeError, match="does not match"):
        FrozenTeacher(dict(teacher=tcfg, teacher_checkpoint=str(bad)))


def test_cross_plane_mil_is_one_sided():
    """§3.6a: a sonar peak above the teacher's confidence is never pulled
    down (gradient on the logits must not be positive)."""
    m0 = M0Mapping(slope=30.75, intercept=-15.375, residual_std=0.5,
                   camera_hfov_deg=30.75, sonar_fov_deg=30.75)
    crit = CrossPlaneMILLoss(m0=m0, bearing_fov_deg=30.75, topk=1)
    obj = torch.full((1, 1, 8, 10), -4.0)
    obj[0, 0, 4, 5] = 4.0                       # p ≈ 0.98 > teacher 0.5
    obj.requires_grad_(True)
    loss, n = crit([[(140.0, 100.0, 180.0, 140.0)]], [torch.tensor([0.5])],
                   [obj], camera_img_w=320)
    assert n.item() == 1
    loss.backward()
    assert obj.grad[0, 0, 4, 5] <= 0             # descent raises the peak
    assert (obj.grad <= 0).all()


def test_cross_plane_mil_nonlinear_bearing_map():
    m0 = M0Mapping(slope=30.75, intercept=-15.375, residual_std=0.1,
                   camera_hfov_deg=30.75, sonar_fov_deg=30.75)
    obj = torch.full((1, 1, 8, 10), -4.0)
    obj[0, 0, 4, 0] = 4.0                        # peak at the left edge
    # a map that sends bearing 0 to column 0: the window lands on the peak
    crit = CrossPlaneMILLoss(m0=m0, bearing_fov_deg=30.75, topk=1,
                             base_margin_deg=0.0,
                             bearing_to_col_norm=lambda th: 0.0)
    loss, _ = crit([[(150.0, 0.0, 170.0, 10.0)]], [torch.tensor([1.0])],
                   [obj], camera_img_w=320)
    assert loss.item() < 0.05


# -- review 2026-11-01 regressions ----------------------------------------

def test_frozen_teacher_requires_checkpoint():
    """#1: no checkpoint -> random teacher (init_weights never reaches
    it), so building one must fail unless explicitly allowed."""
    from aodet.models.teachers.frozen_teacher import FrozenTeacher
    sbb, snc, shd = _stream_cfg()
    tcfg = dict(type="TwoStreamYOLOX", sonar_backbone=sbb, sonar_neck=snc,
                uncoupled_sonar_head=shd)
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        FrozenTeacher(dict(teacher=tcfg))
    FrozenTeacher(dict(teacher=tcfg, allow_random_teacher=True))


def test_clear_teacher_input_does_not_fall_back():
    """#2: teacher_input='clear' without camera_clear must raise, not
    silently feed the degraded view (M2/M3 -> M1)."""
    student = _kd_student()
    data = _batch(b=2, with_clear=False)
    inputs, samples = _prep(student, data, training=True)
    assert inputs["camera_clear"] is None
    student.train()
    with pytest.raises(RuntimeError, match="camera_clear"):
        student.loss(inputs, samples)


def test_teacher_fusion_masks_invalid_camera():
    """#3: the teacher's fusion honours camera_valid, so a masked sample
    equals the no-camera path, as in the teacher's own training."""
    from aodet.models.teachers.frozen_teacher import FrozenTeacher
    sbb, snc, shd = _stream_cfg()
    tcfg = dict(type="TwoStreamYOLOX", sonar_backbone=sbb, sonar_neck=snc,
                camera_backbone=_stream_cfg()[0],
                camera_neck=_stream_cfg()[1], fused_sonar_head=shd,
                fusion=dict(type="CrossAttnFusion",
                            cam_channels=[64, 64, 64],
                            sonar_channels=[64, 64, 64], embed_dim=64,
                            num_heads=4))
    t = FrozenTeacher(dict(teacher=tcfg, allow_random_teacher=True))
    fusion = t.model.fusion
    with torch.no_grad():
        for g in fusion.gamma:
            g.fill_(1.0)               # make the camera term non-trivial
    son = torch.rand(2, 3, *IMG) * 255
    cam = torch.rand(2, 3, *IMG) * 255
    valid = torch.tensor([True, False])
    out = t.forward_head_outputs(cam, son, camera_valid=valid)
    ref = t.forward_head_outputs(None, son)
    full = t.forward_head_outputs(cam, son)
    obj_m, obj_ref, obj_full = (o["sonar"][2][0] for o in (out, ref, full))
    torch.testing.assert_close(obj_m[1], obj_ref[1])
    torch.testing.assert_close(obj_m[0], obj_full[0])
    assert not torch.allclose(obj_full[1], obj_ref[1])


def test_cross_plane_mil_cell_centre_columns():
    """#8: a window centred on bearing 0 maps to the centre cells under
    the (j+0.5)/w convention."""
    m0 = M0Mapping(slope=30.0, intercept=-15.0, residual_std=0.0,
                   camera_hfov_deg=30.0, sonar_fov_deg=30.0)
    crit = CrossPlaneMILLoss(m0=m0, bearing_fov_deg=30.0, topk=1,
                             k_margin=0.0, base_margin_deg=0.0)
    w = 10
    for col in range(w):
        obj = torch.full((1, 1, 4, w), -6.0)
        obj[0, 0, 2, col] = 6.0
        # a camera box whose bearing interval is exactly cell `col`
        x1, x2 = col / w * 320, (col + 1) / w * 320
        loss, n = crit([[(x1, 0.0, x2, 10.0)]], [torch.tensor([1.0])],
                       [obj], camera_img_w=320)
        assert n.item() == 1 and loss.item() < 0.01, col


# -- E11-4 B2b: sonar query dropout on the fusion path -----------------------

def _qdrop_pair(qdrop):
    from aodet.models.fusion import CrossAttnFusion
    torch.manual_seed(0)
    plain = CrossAttnFusion([16, 16], [16, 16], embed_dim=16, num_heads=2,
                            layer_scale_init=0.5)
    torch.manual_seed(0)
    other = CrossAttnFusion([16, 16], [16, 16], embed_dim=16, num_heads=2,
                            layer_scale_init=0.5, sonar_query_dropout=qdrop)
    other.load_state_dict(plain.state_dict())
    son = [torch.randn(4, 16, 6, 5), torch.randn(4, 16, 3, 3)]
    cam = [torch.randn(4, 16, 8, 7), torch.randn(4, 16, 4, 4)]
    return plain, other, cam, son


def test_qdrop_off_is_bit_identical_and_draws_no_rng():
    plain, other, cam, son = _qdrop_pair(None)
    plain.train(), other.train()
    torch.manual_seed(1)
    a = plain(cam, son)
    torch.manual_seed(1)
    b = other(cam, son)
    after = torch.rand(1)
    for x, y in zip(a, b):
        assert torch.equal(x, y)
    torch.manual_seed(1)
    plain(cam, son)
    assert torch.equal(torch.rand(1), after)   # no extra draws either way


def test_qdrop_never_acts_in_eval():
    plain, other, cam, son = _qdrop_pair(dict(p_sample=1.0, keep=0.5))
    plain.eval(), other.eval()
    for x, y in zip(plain(cam, son), other(cam, son)):
        assert torch.equal(x, y)


def test_qdrop_p_zero_is_identity_in_training():
    plain, other, cam, son = _qdrop_pair(dict(p_sample=0.0, keep=0.5))
    plain.train(), other.train()
    for x, y in zip(plain(cam, son), other(cam, son)):
        torch.testing.assert_close(x, y)


def test_qdrop_masks_whole_channels_of_selected_samples_only():
    _, f, _, son = _qdrop_pair(dict(p_sample=0.5, keep=0.5))
    f.train()
    torch.manual_seed(1)   # selects samples 1, 2 only
    out = f(None, son)                      # D-inf path: fused = q exactly
    qs = [proj(s) for proj, s in zip(f.sonar_proj, son)]
    changed = [not torch.equal(out[0][i], qs[0][i]) for i in range(4)]
    assert changed == [False, True, True, False]
    for lvl in range(2):
        for i in range(4):
            if changed[i]:
                o, q = out[lvl][i], qs[lvl][i]
                kept = (o != 0).flatten(1).any(1)
                # kept channels are q / keep, dropped channels are 0
                torch.testing.assert_close(o[kept], q[kept] * 2.0)
                assert torch.all(o[~kept] == 0)
            else:
                assert torch.equal(out[lvl][i], qs[lvl][i])


@pytest.mark.parametrize("bad", [dict(p_sample=1.5, keep=0.5),
                                 dict(p_sample=0.3, keep=0.0)])
def test_qdrop_rejects_bad_args(bad):
    from aodet.models.fusion import CrossAttnFusion
    with pytest.raises(ValueError):
        CrossAttnFusion([16], [16], embed_dim=16, num_heads=2,
                        sonar_query_dropout=bad)
