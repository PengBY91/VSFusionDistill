"""S0-lite, cross-attention fusion: the "does fusion hurt?" probe (Q1).

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
