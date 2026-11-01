"""B4-lite with stop-gradient into the sonar stream (review 2026-10-31 §2.3).

The fusion module sees detached sonar features, so the sonar backbone is
trained by sonar-only supervision and the uncoupled sonar head is
independent of the camera at training time as well as at inference.
Separates "training-time coupling" from "inference-time coupling" in the
RQ-4 branch-correlation readings.

Loss weights (review 2026-11-01 #9): with the fused head's gradient cut
at the fusion input, only the uncoupled head reaches the sonar backbone.
The base 0.5/0.5 split would give that backbone half of B0's supervision,
so the uncoupled head carries B0's full weight here; the fused head keeps
the base 0.5 so its own training matches the non-detached arm.
"""

_base_ = ["s0_cross_attn_yoloxs.py"]

model = dict(
    fusion_detach_sonar=True,
    loss_weights=dict(camera=1.0, fused_sonar=0.5, uncoupled_sonar=1.0),
)
