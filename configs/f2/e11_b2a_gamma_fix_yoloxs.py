"""E11-4 arm B2a: e11_b2 with the fusion LayerScale gamma fixed at the
optimisation level (doc/experiments.md §E11, "E11-4 融合激活臂").

Delta vs e11_b2 (everything else inherited: data, split, D0-only
training, 304-epoch schedule, loss weights, test dataloader on val):

* gamma starts at 0.1 instead of 0 (``layer_scale_init``);
* gamma is exempt from weight decay (``custom_keys['fusion.gamma']``);
  the base recipe only exempts norm and bias, so gamma was decayed
  towards its zero init (rethink §9.2 a).

Tests whether the fusion inertness seen in E09c / E10 M3 is purely an
optimisation problem.  Single seed 0 (single-seed protocol), paired with
B0 seed 0 and B2 seed 0.  Launch gate: E11-2 B2 reads fusion-inert (fused
sonar head val blob AP50 d0 - D-inf within sigma = 0.032).
"""

_base_ = "e11_b2_cross_attn_nominal_yoloxs.py"

model = dict(fusion=dict(layer_scale_init=0.1))

optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys={"fusion.gamma": dict(decay_mult=0.0)}))
