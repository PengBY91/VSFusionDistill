"""E09b: B0 sonar-only, fixed recipe + FIXED sonar range-axis scale jitter.

The E08 redo: AOSonarScaleJitter now runs AFTER the fixed-size resize on
a fixed canvas (crop/pad), so the jitter is no longer undone exactly
(review 2026-10-31 §1.2 -- E08's B0 trained with an effective no-op and
its "scale axis fixed" attribution is retracted as an erratum).  Delta
against E09a is the jitter effect alone; see doc/experiments.md E09 for
the pre-registered criteria.
"""

_base_ = "s0_b0_sonar_only_yoloxs.py"

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
