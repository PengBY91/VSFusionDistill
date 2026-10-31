"""Unit tests for the KD losses (review 2026-10-31 §3.1–§3.3)."""

import pytest

torch = pytest.importorskip("torch")

from aodet.models.losses.branch_kd import (  # noqa: E402
    branch_kd_losses, objectness_kd, score_kd)


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
