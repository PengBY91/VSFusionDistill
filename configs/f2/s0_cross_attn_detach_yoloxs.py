"""B4-lite with stop-gradient into the sonar stream (review 2026-10-31 §2.3).

The fusion module sees detached sonar features, so the sonar backbone is
trained by sonar-only supervision and the uncoupled sonar head is
independent of the camera at training time as well as at inference.
Separates "training-time coupling" from "inference-time coupling" in the
RQ-4 branch-correlation readings.
"""

_base_ = ["s0_cross_attn_yoloxs.py"]

model = dict(fusion_detach_sonar=True)
