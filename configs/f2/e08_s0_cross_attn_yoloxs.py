"""E08: cross-attn two-stream with train-time sonar range-axis scale jitter.

Extends the E07 config (same E07 split, same blob-only training GT);
the only change is `AOSonarScaleJitter` in the train pipeline — see
doc/experiments.md E08 for the pre-registered criteria.
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
