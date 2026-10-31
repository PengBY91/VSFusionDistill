"""S0-lite B4-lite: cross-attn fusion, D0–D4 + D∞ training (Q1 probe).

Naming (review 2026-10-31 §4.4): this config trains with the default
p(s) of AOSampleDegLevel (D0–D4 + D∞), i.e. degradation augmentation +
modality dropout — closest to F2 B4, not the nominal B2 of RQ-0.  See
s0_b2_cross_attn_nominal_yoloxs / s0_b3_cross_attn_degaug_yoloxs.

Two-stream YOLOX-S with all three heads and clean-room cross-attention
fusion (sonar queries camera).  Compare per-branch AP against
s0_b0_sonar_only under the D0–D∞ scan (tools/test.py): if the sonar
branch degrades below B0 as camera input collapses, the S0 answer is
"yes, naive coupling is fragile" and the F2 method space proceeds.
"""

_base_ = [
    "../_base_/models/two_stream_yolox_s.py",
    "../_base_/datasets/paired_uxo.py",
    "../_base_/schedules/schedule_f2.py",
    "../_base_/default_runtime.py",
]
