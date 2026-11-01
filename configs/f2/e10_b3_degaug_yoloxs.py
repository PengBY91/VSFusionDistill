"""E10 B3 arm: degradation-augmented fusion (D0-D4, no D-inf), fixed
recipe + fixed scale jitter + new CrossAttnFusion.

Identical to e10_m3_privileged_yoloxs minus distillation, so M3-vs-B3
isolates the privileged-KD effect (same data, recipe, jitter, split,
seed).  Differs from s0_b3_cross_attn_degaug_yoloxs (the S0 B3) by the
E09b scale jitter on the sonar side; differs from e09_s0_cross_attn
(B4-lite) by p(D-inf)=0: this arm is the *strong baseline* of F2 §2.2
without modality dropout.
"""

_base_ = ["s0_cross_attn_yoloxs.py"]

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    dict(type="AOSonarScaleJitter", scale_range=(0.6, 1.8), prob=1.0),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel",
         probs=dict(d0=0.2, d1=0.2, d2=0.2, d3=0.2, d4=0.2, dinf=0.0)),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
