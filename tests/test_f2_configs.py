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
from mmengine.registry import PARAM_SCHEDULERS, init_default_scope

F2_CONFIGS = [
    "configs/f2/s0_b0_sonar_only_yoloxs.py",
    "configs/f2/s0_cross_attn_yoloxs.py",
    "configs/f2/s0_b2_cross_attn_nominal_yoloxs.py",
    "configs/f2/s0_b3_cross_attn_degaug_yoloxs.py",
    "configs/f2/s0_cross_attn_detach_yoloxs.py",
    "configs/f2/s0_bearing_attn_yoloxs.py",
    "configs/f2/e08_s0_b0_sonar_only_yoloxs.py",
    "configs/f2/e08_s0_cross_attn_yoloxs.py",
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
    if model_cfg.get("fused_sonar_head"):
        total += w["fused_sonar"]
    return total


@pytest.mark.parametrize("path", F2_CONFIGS)
def test_sonar_supervision_weight_matches_b0(path):
    """Review 2026-10-31 §1.3: both sonar heads share the sonar backbone,
    so their weights must sum to B0's single-head weight."""
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
