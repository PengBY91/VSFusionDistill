"""E09c: cross-attn two-stream, fixed recipe + FIXED scale jitter + new
CrossAttnFusion (2D positional encodings, pre-norm, zero-init LayerScale,
camera_valid-masked D-infinity so train and test run the same function --
review 2026-10-31 §2.1/§2.2).

The E08 cross-attn arm finished (2026-10-31 22:07) on the pre-fix code
(no-op jitter, old recipe, old fusion); its numbers are invalid and its
checkpoint cannot load into the new architecture.  This arm redoes it;
primary readouts are fused-vs-uncoupled gap and D-inf behaviour, not the
domain question (see doc/experiments.md E09).
"""

_base_ = "s0_cross_attn_yoloxs.py"

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    # after the resize: a fixed-size resize placed after the jitter undoes
    # it exactly (review 2026-10-31 §1.2)
    dict(type="AOSonarScaleJitter", scale_range=(0.6, 1.8), prob=1.0),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel"),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
