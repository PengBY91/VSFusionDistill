"""S0-lite, B0: sonar-only baseline (F2 method space B0).

The strongest sonar-only reference: same YOLOX-S sonar stream, one head,
camera stream entirely absent.  All fused/privileged methods must beat
B0's sonar AP on clean water to justify the fusion machinery (Q1), and
B0 is the fixed reference for ΔP_miss at every degradation level.
"""

_base_ = [
    "../_base_/models/two_stream_yolox_s.py",
    "../_base_/datasets/paired_uxo.py",
    "../_base_/schedules/schedule_f2.py",
    "../_base_/default_runtime.py",
]

model = dict(
    camera_backbone=None,
    camera_neck=None,
    camera_head=None,
    fused_sonar_head=None,
    # _delete_: replace the base CrossAttnFusion dict, not merge into it
    fusion=dict(type="NoneFusion", _delete_=True),
)
