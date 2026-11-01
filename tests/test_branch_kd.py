"""Unit tests for the KD losses (review 2026-10-31 §3.1–§3.3)."""

import pytest

torch = pytest.importorskip("torch")

from aodet.models.losses.branch_kd import (  # noqa: E402
    branch_kd_losses, objectness_kd, score_kd)


def _branch(cls, obj):
    return ([cls], [torch.zeros(cls.shape[0], 4, *cls.shape[-2:])], [obj])


def _one_fg(size=20, fg=(10, 10)):
    t = torch.full((1, 1, size, size), -8.0)
    t[0, 0, fg[0], fg[1]] = 8.0
    return t


def test_objectness_kd_foreground_dominates():
    t = [_one_fg()]
    miss_fg = [torch.full((1, 1, 20, 20), -8.0)]          # misses the target
    extra_fp = [t[0].clone()]
    extra_fp[0][0, 0, 0, 0] = 8.0                          # one false alarm
    assert objectness_kd(miss_fg, t) > 10 * objectness_kd(extra_fp, t)


def test_objectness_kd_scale_is_not_diluted():
    """One target in an 80x80 map: the old dense MSE was ~1e-4 here; the
    weighted loss stays O(0.1–1) like YOLOX's own obj loss."""
    t = [_one_fg(80, (40, 40))]
    s = [torch.zeros(1, 1, 80, 80)]
    assert objectness_kd(s, t) > 0.1


def test_objectness_kd_valid_mask_and_zero_at_match():
    t = [torch.cat([_one_fg(), _one_fg()])]
    assert objectness_kd([t[0].clone()], t) < 0.01         # matched -> ~0
    s = [torch.cat([t[0][:1].clone(), torch.zeros(1, 1, 20, 20)])]
    masked = objectness_kd(s, t, valid=torch.tensor([True, False]))
    assert masked < 0.01                                    # bad sample masked


def test_score_kd_ignores_background_class_logits():
    """Background class logits are never supervised in YOLOX: changing
    them where the teacher sees nothing barely moves the loss."""
    t_obj = [_one_fg()]
    t_cls = [torch.zeros(1, 3, 20, 20)]
    t_cls[0][0, 1, 10, 10] = 6.0
    s_obj = [t_obj[0].clone()]
    s_cls_a = [t_cls[0].clone()]
    s_cls_b = [t_cls[0].clone()]
    s_cls_b[0][:, :, :5, :5] = torch.randn(1, 3, 5, 5) * 5  # background noise
    a = score_kd(s_cls_a, s_obj, t_cls, t_obj)
    b = score_kd(s_cls_b, s_obj, t_cls, t_obj)
    assert abs(float(b - a)) < 0.05
    s_cls_c = [t_cls[0].clone()]
    s_cls_c[0][0, 1, 10, 10] = -6.0                         # wrong at target
    assert score_kd(s_cls_c, s_obj, t_cls, t_obj) > a + 0.05


def test_branch_kd_routes_and_errors():
    out = (torch.zeros(1, 3, 4, 4),), (torch.zeros(1, 4, 4, 4),), \
        (torch.zeros(1, 1, 4, 4),)
    student = {"sonar_uncoupled": tuple(o for o in out)}
    teacher = {"sonar": tuple(o for o in out)}
    losses = branch_kd_losses(student, teacher,
                              cfg=dict(routes={"sonar_uncoupled": "sonar"}))
    assert set(losses) == {"sonar_uncoupled_obj"}
    with pytest.raises(RuntimeError):
        branch_kd_losses(student, {"camera": teacher["sonar"]},
                         cfg=dict(routes={"sonar_uncoupled": "sonar"}))


# -- review 2026-11-01 regressions ----------------------------------------

def test_kd_terms_are_exactly_zero_at_match():
    """#6: KL form, so a perfect match is 0, not the teacher entropy."""
    t_obj = [torch.randn(2, 1, 8, 8)]
    t_cls = [torch.randn(2, 3, 8, 8)]
    assert abs(float(objectness_kd([t_obj[0].clone()], t_obj))) < 1e-5
    assert abs(float(score_kd([t_cls[0].clone()], [t_obj[0].clone()],
                              t_cls, t_obj))) < 1e-4


def test_score_kd_runs_under_autocast():
    """#7: no F.binary_cross_entropy inside an autocast region."""
    t_obj, t_cls = [_one_fg()], [torch.zeros(1, 3, 20, 20)]
    with torch.autocast("cpu", dtype=torch.bfloat16):
        v = score_kd([t_cls[0].clone().requires_grad_()],
                     [torch.zeros(1, 1, 20, 20, requires_grad=True)],
                     t_cls, t_obj)
    assert v.dtype == torch.float32 and torch.isfinite(v)


def test_reach_gates_only_uncoupled_route_and_score_term():
    """#4: reach applies to the sonar-only head's obj AND score terms, and
    never to the fused head (which sees the camera)."""
    t_obj, t_cls = _one_fg(), torch.zeros(1, 3, 20, 20)
    t_cls[0, 1, 10, 10] = 6.0
    s_obj = torch.full((1, 1, 20, 20), -8.0)        # student misses it
    s_cls = torch.zeros(1, 3, 20, 20)
    teacher = {"sonar": _branch(t_cls, t_obj)}
    student = {"sonar": _branch(s_cls, s_obj),
               "sonar_uncoupled": _branch(s_cls.clone(), s_obj.clone())}
    cfg = dict(routes={"sonar": "sonar", "sonar_uncoupled": "sonar"},
               score_weight=1.0)
    unreachable = [torch.zeros(1, 1, 20, 20)]     # camera-only target
    free = branch_kd_losses(student, teacher, cfg=cfg)
    gated = branch_kd_losses(student, teacher, cfg=cfg, reach=unreachable)
    for k in ("sonar_obj", "sonar_score"):
        torch.testing.assert_close(gated[k], free[k])
    for k in ("sonar_uncoupled_obj", "sonar_uncoupled_score"):
        assert gated[k] < 0.2 * free[k], (k, gated[k], free[k])


def test_reach_keeps_background_floor():
    """#5: reach must not erase the background push-down weight."""
    t = [torch.full((1, 1, 20, 20), -8.0)]           # no target anywhere
    s = [t[0].clone()]
    s[0][0, 0, 0, 0] = 8.0                           # one false alarm
    zero = [torch.zeros(1, 1, 20, 20)]
    torch.testing.assert_close(objectness_kd(s, t, reach=zero),
                               objectness_kd(s, t))


# -- distillation temperature (F2 §5.3; E10) -------------------------------

def test_tau_one_is_bitwise_identity():
    """tau=1 (explicit) must reproduce the untempered loss exactly."""
    t_obj = [torch.randn(2, 1, 8, 8)]
    t_cls = [torch.randn(2, 3, 8, 8)]
    s_obj = [torch.randn(2, 1, 8, 8)]
    s_cls = [torch.randn(2, 3, 8, 8)]
    torch.testing.assert_close(
        objectness_kd(s_obj, t_obj, tau=1.0), objectness_kd(s_obj, t_obj))
    torch.testing.assert_close(
        score_kd(s_cls, s_obj, t_cls, t_obj, tau=1.0),
        score_kd(s_cls, s_obj, t_cls, t_obj))


def test_tau_softens_teacher_targets():
    """tau changes how hard the teacher's saturated confidences press on a
    partially-matching student.  Teacher logit 8 (sigma .9997), student
    logit 4 (sigma .982): both taus give a positive finite KL on this
    fixed mismatched pair, and the tau=2 value differs from tau=1 (the
    optimum stays 'copy the teacher'; tau reshapes curvature, not the
    argmin -- which is why zero-at-match holds separately)."""
    t = [torch.full((1, 1, 8, 8), 8.0)]
    partial = [torch.full((1, 1, 8, 8), 4.0)]
    at_tau1 = float(objectness_kd(partial, t, tau=1.0))
    at_tau2 = float(objectness_kd(partial, t, tau=2.0))
    assert 0.0 < at_tau1 < 1.0               # hard target forgives less
    assert 0.0 < at_tau2 < 4.0
    assert abs(at_tau2 - at_tau1) > 1e-3     # tau actually changes the loss


def test_tau_keeps_zero_loss_at_match():
    """s == t must be a perfect match at any tau (both logits divided)."""
    t_obj = [torch.randn(2, 1, 8, 8)]
    t_cls = [torch.randn(2, 3, 8, 8)]
    for tau in (2.0, 4.0):
        assert abs(float(objectness_kd([t_obj[0].clone()], t_obj,
                                       tau=tau))) < 1e-5
        assert abs(float(score_kd([t_cls[0].clone()], [t_obj[0].clone()],
                                  t_cls, t_obj, tau=tau))) < 1e-4


def test_tau_scales_gradient_not_just_value():
    """Hinton scaling: d(loss)/d(student logit) keeps O(1) magnitude across
    tau (the raw /tau softmax-style gradient would shrink like 1/tau)."""
    t = [torch.full((1, 1, 8, 8), 6.0)]
    g = {}
    for tau in (1.0, 4.0):
        s = [torch.zeros(1, 1, 8, 8, requires_grad=True)]
        objectness_kd(s, t, tau=tau).backward()
        g[tau] = float(s[0].grad.abs().sum())
    assert 0.2 < g[4.0] / g[1.0] < 5.0


def test_tau_flows_through_branch_kd_cfg():
    t_obj = [torch.full((1, 1, 8, 8), 6.0)]
    s_obj = [torch.zeros(1, 1, 8, 8)]                # student misses it
    cls = [torch.zeros(1, 3, 8, 8)]
    student = {"sonar_uncoupled": _branch(cls[0], s_obj[0])}
    teacher = {"sonar": _branch(cls[0], t_obj[0])}
    a = branch_kd_losses(student, teacher, cfg=dict(tau=1.0))
    b = branch_kd_losses(student, teacher, cfg=dict(tau=2.0))
    assert set(a) == {"sonar_uncoupled_obj"}
    assert float(b["sonar_uncoupled_obj"]) > float(a["sonar_uncoupled_obj"]) > 0
    with pytest.raises(ValueError):
        objectness_kd(s_obj, t_obj, tau=0.0)
