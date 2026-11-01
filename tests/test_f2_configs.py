"""Config-level tests for the f2 S0-lite configs.

The synthetic e2e smoke (test_tools_e2e) runs its own minimal config, so
base-config bugs can slip through until real-data training -- exactly how
the schedule_f2 convert_to_iter_based-on-an-iter-based-LinearLR bug
survived W3.  These tests load the real configs and build the pieces
Runner.from_cfg would build.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aodet.datasets  # noqa: F401  (registry side effects)
import aodet.models  # noqa: F401
from mmengine.config import Config
from mmengine.registry import init_default_scope

# QuadraticWarmupLR registers into mmdet's registry root (Runner resolves
# schedulers through default_scope='mmdet'), not mmengine's
from mmdet.registry import PARAM_SCHEDULERS

F2_CONFIGS = [
    "configs/f2/s0_b0_sonar_only_yoloxs.py",
    "configs/f2/s0_cross_attn_yoloxs.py",
    "configs/f2/s0_b2_cross_attn_nominal_yoloxs.py",
    "configs/f2/s0_b3_cross_attn_degaug_yoloxs.py",
    "configs/f2/s0_cross_attn_detach_yoloxs.py",
    "configs/f2/s0_bearing_attn_yoloxs.py",
    "configs/f2/e08_s0_b0_sonar_only_yoloxs.py",
    "configs/f2/e08_s0_cross_attn_yoloxs.py",
    "configs/f2/e09_s0_b0_recipe_yoloxs.py",
    "configs/f2/e09_s0_b0_jitter_yoloxs.py",
    "configs/f2/e09_s0_cross_attn_yoloxs.py",
    # E10 (M3 itself is content-checked separately: building it requires
    # the teacher checkpoint to exist)
    "configs/f2/e10_t_b2_yoloxl_teacher.py",
    "configs/f2/e10_b3_degaug_yoloxs.py",
    # E11 (the three distillation students are content-checked separately,
    # same reason as M3)
    "configs/f2/e11_t_b2s_nominal_yoloxs.py",
]


def test_param_scheduler_builds():
    """Each scheduler entry builds as Runner would (epoch_length and
    optimizer passed as default_args by the runner; convert_to_iter_based
    is only legal on epoch-based entries)."""
    import torch

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.01)
    for path in F2_CONFIGS:
        cfg = Config.fromfile(path)
        schedulers = [
            PARAM_SCHEDULERS.build(dict(s, epoch_length=275),
                                   default_args=dict(optimizer=opt))
            for s in cfg.param_scheduler
        ]
        assert len(schedulers) == len(cfg.param_scheduler)


@pytest.mark.parametrize("path", F2_CONFIGS)
def test_model_builds_from_config(path):
    from mmdet.registry import MODELS

    init_default_scope("mmdet")
    cfg = Config.fromfile(path)
    model = MODELS.build(cfg.model)
    assert model is not None


def _sonar_weight(model_cfg):
    w = model_cfg.loss_weights
    total = 0.0
    if model_cfg.get("uncoupled_sonar_head"):
        total += w["uncoupled_sonar"]
    # a detached fusion input stops the fused head's gradient before the
    # sonar backbone (review 2026-11-01 #9)
    if model_cfg.get("fused_sonar_head") and \
            not model_cfg.get("fusion_detach_sonar", False):
        total += w["fused_sonar"]
    return total


@pytest.mark.parametrize("path", F2_CONFIGS)
def test_sonar_supervision_weight_matches_b0(path):
    """Review 2026-10-31 §1.3: the sonar-head weights that reach the sonar
    backbone must sum to B0's single-head weight."""
    b0 = Config.fromfile(F2_CONFIGS[0]).model
    assert _sonar_weight(Config.fromfile(path).model) == \
        pytest.approx(_sonar_weight(b0))


def test_yolox_recipe():
    """Review §1.4: COCO-pretrained streams, YOLOX lr scaling, no decay on
    norm/bias, official test thresholds."""
    cfg = Config.fromfile(F2_CONFIGS[1])
    for key in ("sonar_backbone", "sonar_neck", "camera_backbone",
                "camera_neck"):
        assert cfg.model[key].init_cfg.type == "Pretrained", key
    assert cfg.optim_wrapper.optimizer.lr == pytest.approx(0.01 * 8 / 64)
    assert cfg.optim_wrapper.paramwise_cfg.norm_decay_mult == 0.0
    assert cfg.model.test_cfg.score_thr == 0.01
    assert cfg.model.test_cfg.nms.iou_threshold == 0.65
    assert any(h.type == "EMAHook" for h in cfg.custom_hooks)


def test_b2_trains_nominal_only():
    cfg = Config.fromfile("configs/f2/s0_b2_cross_attn_nominal_yoloxs.py")
    deg = [t for t in cfg.train_dataloader.dataset.pipeline
           if t["type"] == "AOSampleDegLevel"][0]
    assert deg["probs"]["d0"] == 1.0


# -- E10: privileged distillation mainline ----------------------------------

E10_TEACHER = "configs/f2/e10_t_b2_yoloxl_teacher.py"
E10_M3 = "configs/f2/e10_m3_privileged_yoloxs.py"


def _deg_probs(cfg):
    return [t for t in cfg.train_dataloader.dataset.pipeline
            if t["type"] == "AOSampleDegLevel"][0]["probs"]


def _has_jitter(cfg):
    return any(t["type"] == "AOSonarScaleJitter"
               for t in cfg.train_dataloader.dataset.pipeline)


def test_e10_teacher_is_big_nominal_and_jittered():
    """Gate-1 inputs: L-capacity streams, COCO-pretrained init, nominal D0,
    sonar scale jitter (a scale-locked teacher would be a weak teacher)."""
    cfg = Config.fromfile(E10_TEACHER)
    assert cfg.model.sonar_neck.in_channels == [256, 512, 1024]
    assert cfg.model.sonar_neck.out_channels == 256
    assert cfg.model.camera_backbone.deepen_factor == 1.0
    for key in ("sonar_backbone", "sonar_neck", "camera_backbone",
                "camera_neck"):
        assert cfg.model[key].init_cfg.type == "Pretrained", key
        assert "yolox_l" in cfg.model[key].init_cfg.checkpoint, key
    assert _deg_probs(cfg)["d0"] == 1.0
    assert _has_jitter(cfg)


def test_e10_b3_degaug_matches_m3_minus_kd():
    """M3's fairness anchor: B3 = same recipe/data/jitter, D0-D4 no D-inf."""
    cfg = Config.fromfile("configs/f2/e10_b3_degaug_yoloxs.py")
    probs = _deg_probs(cfg)
    assert probs["dinf"] == 0.0
    assert all(abs(probs[f"d{i}"] - 0.2) < 1e-9 for i in range(5))
    assert _has_jitter(cfg)
    m3 = Config.fromfile(E10_M3)
    assert _deg_probs(m3) == probs
    assert _has_jitter(m3)


def test_e10_m3_teacher_dict_matches_teacher_config():
    """The embedded teacher must equal the teacher config's model (guards
    copy drift between the two files)."""
    from mmengine.config import ConfigDict

    teacher = Config.fromfile(E10_TEACHER).model
    m3 = Config.fromfile(E10_M3).model.distillation.teacher
    a = ConfigDict(teacher)
    b = ConfigDict(m3)
    assert a.keys() == b.keys(), set(a) ^ set(b)
    for k in a:
        assert a[k] == b[k], k


def test_e10_m3_kd_settings():
    """Pre-registered KD settings: clear-view teacher (privileged), obj +
    score routes, tau=2, no reach (§5.4: validate the proxy before use),
    no feature KD (later ablation), checkpoint required."""
    kd = Config.fromfile(E10_M3).model.distillation
    assert kd.teacher_input == "clear"
    assert kd.obj_weight == 1.0
    assert kd.score_weight == 1.0
    assert kd.tau == 2.0
    assert kd.routes == {"camera": "camera", "sonar": "sonar",
                         "sonar_uncoupled": "sonar"}
    assert kd.teacher_checkpoint.endswith("epoch_100.pth")
    assert "reach" not in kd
    assert float(kd.get("feat_weight", 0.0)) == 0.0
    assert kd.teacher.type == "TwoStreamYOLOX"


def test_e10_m3_teacher_builds():
    """The embedded teacher model dict constructs (checked standalone: the
    full M3 build needs the trained teacher checkpoint to exist)."""
    from mmdet.registry import MODELS
    init_default_scope("mmdet")
    teacher = Config.fromfile(E10_M3).model.distillation.teacher
    model = MODELS.build(teacher)
    assert model.camera_backbone is not None


# -- E11: 2x2 privilege/capacity attribution (M1 / M2 / D-self) --------------

E11_TEACHER_S = "configs/f2/e11_t_b2s_nominal_yoloxs.py"
E11_M1 = "configs/f2/e11_m1_capacity_yoloxs.py"
E11_M2 = "configs/f2/e11_m2_privilege_yoloxs.py"
E11_DSELF = "configs/f2/e11_dself_yoloxs.py"


def test_e11_teacher_s_is_nominal_jittered_and_small():
    """M2's teacher: the T-L recipe at S capacity (nominal D0, E09b sonar
    jitter, S geometry) -- differs from B3 only in the degradation probs."""
    cfg = Config.fromfile(E11_TEACHER_S)
    assert cfg.model.sonar_neck.in_channels == [128, 256, 512]
    assert cfg.model.sonar_neck.out_channels == 128
    assert cfg.model.camera_backbone.deepen_factor == 0.33
    assert _deg_probs(cfg)["d0"] == 1.0
    assert _has_jitter(cfg)


def test_e11_students_share_b3_recipe():
    """All three 2x2 students = B3 exactly (same data/recipe/jitter), so
    only the teacher side varies across the 2x2."""
    probs = _deg_probs(Config.fromfile("configs/f2/e10_b3_degaug_yoloxs.py"))
    for path in (E11_M1, E11_M2, E11_DSELF):
        cfg = Config.fromfile(path)
        assert _deg_probs(cfg) == probs, path
        assert _has_jitter(cfg), path


def test_e11_2x2_kd_matrix():
    """The 2x2 cells differ only in teacher checkpoint + teacher_input;
    every other KD setting equals M3's (tau=2, obj+score, same routes,
    no reach, no feature KD)."""
    m3 = Config.fromfile(E10_M3).model.distillation
    want = {
        E11_M1: ("degraded", "e10_t_b2_yoloxl_teacher_seed0"),
        E11_M2: ("clear", "e11_t_b2s_nominal_yoloxs_seed0"),
        E11_DSELF: ("degraded", "e10_b3_degaug_yoloxs_seed0"),
    }
    for path, (t_in, ckpt_frag) in want.items():
        kd = Config.fromfile(path).model.distillation
        assert kd.teacher_input == t_in, path
        assert ckpt_frag in kd.teacher_checkpoint, path
        assert kd.obj_weight == m3.obj_weight == 1.0, path
        assert kd.score_weight == m3.score_weight == 1.0, path
        assert kd.tau == m3.tau == 2.0, path
        assert kd.bg_floor == m3.bg_floor, path
        assert kd.routes == m3.routes, path
        assert "reach" not in kd, path
        assert float(kd.get("feat_weight", 0.0)) == 0.0, path


def test_e11_m1_inherits_m3_teacher_unchanged():
    """M1 = M3 minus privilege: base is the M3 config, only
    teacher_input flips, so the embedded teacher must be identical."""
    m1 = Config.fromfile(E11_M1).model.distillation
    m3 = Config.fromfile(E10_M3).model.distillation
    assert m1.teacher == m3.teacher
    assert m1.teacher_checkpoint == m3.teacher_checkpoint


def test_e11_m2_and_dself_teacher_dicts_match_source_models():
    """The embedded teachers must equal their source configs' models
    (guards copy drift): M2's == the E11 S teacher config, D-self's ==
    B3's own model."""
    from mmengine.config import ConfigDict

    sources = {
        E11_M2: Config.fromfile(E11_TEACHER_S).model,
        E11_DSELF: Config.fromfile("configs/f2/e10_b3_degaug_yoloxs.py").model,
    }
    for path, src in sources.items():
        kd = Config.fromfile(path).model.distillation.teacher
        a, b = ConfigDict(src), ConfigDict(kd)
        assert a.keys() == b.keys(), (path, set(a) ^ set(b))
        for k in a:
            assert a[k] == b[k], (path, k)


def test_e11_s_teacher_builds():
    """The embedded S teacher model dict constructs standalone."""
    from mmdet.registry import MODELS
    init_default_scope("mmdet")
    teacher = Config.fromfile(E11_M2).model.distillation.teacher
    model = MODELS.build(teacher)
    assert model.camera_backbone is not None
