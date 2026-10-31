"""S0-lite B3: cross-attn fusion trained with D0–D4 degradation, no D∞.

F2 §2.2 B3 (the main strong baseline): degradation-augmented fusion
without modality dropout.  Read against B2 (nominal) for "how much does
seeing degradation in training recover", and against B4-lite
(s0_cross_attn_yoloxs, D0–D4 + D∞) for the modality-dropout increment.
"""

_base_ = ["s0_cross_attn_yoloxs.py"]

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel",
         probs=dict(d0=0.2, d1=0.2, d2=0.2, d3=0.2, d4=0.2, dinf=0.0)),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
