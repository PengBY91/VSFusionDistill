"""Two-stream YOLOX-S, three heads (F2 §5.1; setup doc W3).

Both streams are YOLOX-S-width (widen 0.5 -> neck out 128).  The base
configures the *fused* topology (cross-attention, all three heads);
B0-style configs null out the camera stream / heads by overriding keys.

``num_classes``: set to the converted dataset's category count (UXO
categories are fixed by aodet.datasets.converters.uxo_to_paired once the
sample archive is verified — TODO 14.2; 1 = single UXO class).
"""

num_classes = 1

# YOLOX-S building blocks (mmdet 3.3 yolox_s geometry)
_backbone = dict(
    type="CSPDarknet",
    deepen_factor=0.33,
    widen_factor=0.5,
    out_indices=(2, 3, 4),
)
_neck = dict(
    type="YOLOXPAFPN",
    in_channels=[128, 256, 512],
    out_channels=128,
    num_csp_blocks=1,
)
_head = dict(
    type="YOLOXHead",
    num_classes=num_classes,
    in_channels=128,
    feat_channels=128,
)

model = dict(
    type="TwoStreamYOLOX",
    data_preprocessor=dict(
        type="PairedDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    sonar_backbone=_backbone,
    sonar_neck=_neck,
    camera_backbone=_backbone,
    camera_neck=_neck,
    camera_head=_head,
    uncoupled_sonar_head=_head,
    fused_sonar_head=_head,
    fusion=dict(
        type="CrossAttnFusion",
        cam_channels=[128, 128, 128],
        sonar_channels=[128, 128, 128],
        embed_dim=128,
        num_heads=4,
    ),
    loss_weights=dict(camera=1.0, fused_sonar=1.0, uncoupled_sonar=1.0),
    train_cfg=dict(
        assigner=dict(
            type="SimOTAAssigner",
            center_radius=2.5,
            candidate_topk=10,
        ),
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type="nms", iou_threshold=0.55),
        max_per_img=100,
    ),
)
