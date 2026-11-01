"""E11 M2 teacher: YOLOX-S two-stream fusion, nominal (D0) clear camera.

F2 §2.2 M2's "same-capacity fusion teacher that sees the clear view" --
the T-L recipe (e10_t_b2_yoloxl_teacher) at student capacity: nominal D0
training + the E09b sonar scale jitter, both streams COCO-pretrained
YOLOX-S.  Differs from B3 (e10_b3_degaug_yoloxs) ONLY in the degradation
probs (no D0-D4 augmentation), so at equal capacity the teacher-vs-B3
contrast isolates clear-view training for the E11 2x2.

Its checkpoint feeds e11_m2_privilege_yoloxs via
``distillation.teacher_checkpoint`` (no random teachers -- review
2026-11-01 #1).  Keep this model in sync with the teacher block in that
config; tests/test_f2_configs.py asserts the two model dicts match.  No
AMP: only the L teacher needed AmpOptimWrapper (S fp32 fits the card).
"""

_base_ = ["s0_b2_cross_attn_nominal_yoloxs.py"]

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=(640, 640), sonar_size=(640, 640), keep_ratio=False),
    dict(type="AOSonarScaleJitter", scale_range=(0.6, 1.8), prob=1.0),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel",
         probs=dict(d0=1.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0, dinf=0.0)),
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
