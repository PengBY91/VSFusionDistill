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
