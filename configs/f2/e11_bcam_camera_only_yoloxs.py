"""E11 B-class amendment 4: Bcam camera-only on the full balanced UXO
split (seeds 0/1/2).

The unimodal camera reference of the E11 platform (ledger E11,
pre-registration amendment B4, adopted 2026-11-03).  Identical to
e11_b2 in data, pipeline, recipe and schedule by inheritance -- only
the sonar heads, the fusion module and the sonar supervision are
removed, so B2's camera head can be compared against a same-recipe
unimodal camera model (Delta_fuse_cam), and the class-transplant
upper bound can draw camera class distributions from an unfused
source.  Not a mainline gate: no E11-2/E11-3 verdict depends on it.

Model note: TwoStreamYOLOX keeps the sonar backbone built (paired
batches always carry sonar images) but skips its forward when both
sonar heads are None (aodet/models/detectors/two_stream_yolox.py);
training losses carry only the camera/* keys.

Degradation applies to the camera input only, so the val D0-Dinf scan
is the unimodal camera degradation curve; training stays nominal D0
(same probs as B2, inherited).  Evaluation reads val only (test stays
sealed); every tools/test.py invocation passes an explicit --split.
"""

_base_ = "e11_b2_cross_attn_nominal_yoloxs.py"

model = dict(
    uncoupled_sonar_head=None,
    fused_sonar_head=None,
    fusion=dict(type="NoneFusion", _delete_=True),
    loss_weights=dict(camera=1.0, fused_sonar=0.0, uncoupled_sonar=0.0),
)
