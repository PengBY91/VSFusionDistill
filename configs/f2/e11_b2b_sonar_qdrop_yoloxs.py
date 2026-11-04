"""E11-4 arm B2b: B2a plus sonar query dropout on the fusion path
(doc/experiments.md §E11, "E11-4 融合激活臂").

Delta vs e11_b2a: in training, each sample is selected with p = 0.3 and
its projected sonar features in CrossAttnFusion get a per-channel
Bernoulli(0.5) mask (scaled by 1/0.5).  Only the fused sonar head sees
the dropped features; the uncoupled sonar head and the backbones run
unchanged, and evaluation never drops.

Tests whether adding pressure to use the camera (B0 fits train with
sonar alone, self-eval 1.0) lets the fused head carry the camera's class
information.  Single seed 0, paired with B0 / B2 / B2a seed 0; same
launch gate as B2a.
"""

_base_ = "e11_b2a_gamma_fix_yoloxs.py"

model = dict(fusion=dict(
    sonar_query_dropout=dict(p_sample=0.3, keep=0.5)))
