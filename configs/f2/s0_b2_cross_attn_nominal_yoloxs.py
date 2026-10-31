"""S0-lite B2: cross-attn fusion trained on nominal (D0) camera only.

RQ-0's direct object (F2 §3 S0): a fusion model that never saw camera
degradation in training, scanned D0–D∞ at inference.  The pre-existing
s0_cross_attn_yoloxs trains with D0–D4 + D∞ (B4-lite), which is built to
ignore a failing camera and therefore biased towards "no mutual
poisoning" (review 2026-10-31 §4.4).
"""

_base_ = ["s0_cross_attn_yoloxs.py"]

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel",
         probs=dict(d0=1.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0, dinf=0.0)),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
