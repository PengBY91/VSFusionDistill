"""B4-lite with bearing-axial cross-attention (review 2026-10-31 §2.2 B).

Sonar positions attend to camera *columns* (camera features are
attention-pooled over the vertical axis): the "camera column -> sonar
bearing, range unknown" correspondence of F2 张力 4, at O(HsWs·Wc) cost.
No calibrated bearing tables yet (dataset-level tables are a callable
argument; per-recording UXO calibration is not wired), so the mapping is
learned through 1D bearing-axis positional encodings.
"""

_base_ = ["s0_cross_attn_yoloxs.py"]

model = dict(fusion=dict(
    _delete_=True,
    type="BearingAxialFusion",
    cam_channels=[128, 128, 128],
    sonar_channels=[128, 128, 128],
    embed_dim=128,
    num_heads=4,
    layer_scale_init=0.0,
))
