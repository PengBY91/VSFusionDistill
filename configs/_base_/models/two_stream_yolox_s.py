"""Two-stream YOLOX-S, three heads (F2 §5.1; setup doc W3).

Both streams are YOLOX-S-width (widen 0.5 -> neck out 128).  The base
configures the *fused* topology (cross-attention, all three heads);
B0-style configs null out the camera stream / heads by overriding keys.

``num_classes``: the UXO category count verified on sample.7z and fixed
by aodet.datasets.converters.uxo_to_paired.CATEGORIES (3 classes:
100lbs_aircraft_bomb, incindiary_grenade, mortar_shell).
"""

num_classes = 3

# COCO-pretrained YOLOX-S: the checkpoint the W1 reproduction gate (E01,
# AP 40.30) validated.  Both streams start from it (setup doc §4.2; review
# 2026-10-31 §1.4) — training ~9M/20M parameters from scratch on ~1.7k
# frames left cross-recording generalisation unattributable.  Heads stay
# randomly initialised (different class count).  init_cfg only fires in
# Runner's init_weights(); tools/test.py loads full checkpoints on top.
_yolox_s_ckpt = ("https://download.openmmlab.com/mmdetection/v2.0/yolox/"
                 "yolox_s_8x8_300e_coco/"
                 "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth")

# YOLOX-S building blocks (mmdet 3.3 yolox_s geometry)
_backbone = dict(
    type="CSPDarknet",
    deepen_factor=0.33,
    widen_factor=0.5,
    out_indices=(2, 3, 4),
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_s_ckpt,
                  prefix="backbone."),
)
_neck = dict(
    type="YOLOXPAFPN",
    in_channels=[128, 256, 512],
    out_channels=128,
    num_csp_blocks=1,
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_s_ckpt,
                  prefix="neck."),
)
_head = dict(
    type="YOLOXHead",
    num_classes=num_classes,
    in_channels=128,
    feat_channels=128,
)

model = dict(
    type="TwoStreamYOLOX",
    # YOLOX convention: raw 0-255 BGR inputs, no mean/std normalization
    # (mean/std None disables normalize in DetDataPreprocessor).  Verified
    # the hard way on the W1 COCO reproduction — ImageNet-style normalize
    # collapses the YOLOX objectness branch and AP to ~0; it also breaks
    # any future init of the camera stream from COCO-pretrained YOLOX-S.
    # Paired streams are resized to exact 640x640 upstream, so
    # pad_size_divisor never actually pads.
    data_preprocessor=dict(
        type="PairedDataPreprocessor",
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
        # zero-init LayerScale: the fused head starts as a sonar-only head
        layer_scale_init=0.0,
        pos_embed=True,
    ),
    # the two sonar heads share the sonar backbone: their weights sum to
    # B0's single sonar head so Δ_s is not confounded by doubled sonar
    # supervision (review 2026-10-31 §1.3)
    loss_weights=dict(camera=1.0, fused_sonar=0.5, uncoupled_sonar=0.5),
    train_cfg=dict(
        assigner=dict(
            type="SimOTAAssigner",
            center_radius=2.5,
            candidate_topk=10,
        ),
    ),
    # official YOLOX test thresholds (E01: score 0.01, NMS 0.65)
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type="nms", iou_threshold=0.65),
        max_per_img=100,
    ),
)
