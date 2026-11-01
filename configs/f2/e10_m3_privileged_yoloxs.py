"""E10 M3 arm: privileged distillation -- big clear-view fusion teacher
into a D0-D4 degraded fusion student (F2 §2.2 M3, mainline).

Student = e10_b3_degaug_yoloxs exactly (same recipe / data / jitter /
split / seed); the ONLY difference is the frozen YOLOX-L teacher that
always sees the clear camera (teacher_input='clear') and the KD losses:

* routes: teacher camera head -> student camera head (the privileged
  term under degradation), teacher fused sonar head -> both student
  sonar heads (the CMOLD-style cross-modal enhancement);
* obj + score terms, tau=2 (F2 §5.3 temperature; E10 pre-registration),
  no feature KD and no reach for the first run -- §5.4 requires the
  reachability proxy to be validated against errors *before* any
  intervention, and feature KD is a later ablation.

The teacher model dict below must stay in sync with
e10_t_b2_yoloxl_teacher.py; tests/test_f2_configs.py asserts equality,
and teacher_checkpoint is required (review 2026-11-01 #1 -- no random
teachers).
"""

_base_ = ["e10_b3_degaug_yoloxs.py"]

_yolox_l_ckpt = ("https://download.openmmlab.com/mmdetection/v2.0/yolox/"
                 "yolox_l_8x8_300e_coco/"
                 "yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth")

_backbone_l = dict(
    type="CSPDarknet",
    deepen_factor=1.0,
    widen_factor=1.0,
    out_indices=(2, 3, 4),
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_l_ckpt,
                  prefix="backbone."),
)
_neck_l = dict(
    type="YOLOXPAFPN",
    in_channels=[256, 512, 1024],
    out_channels=256,
    num_csp_blocks=3,
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_l_ckpt,
                  prefix="neck."),
)
_head_l = dict(
    type="YOLOXHead",
    num_classes=3,
    in_channels=256,
    feat_channels=256,
)

model = dict(
    distillation=dict(
        teacher=dict(
            type="TwoStreamYOLOX",
            data_preprocessor=dict(
                type="PairedDataPreprocessor",
                pad_size_divisor=32,
            ),
            sonar_backbone=_backbone_l,
            sonar_neck=_neck_l,
            camera_backbone=_backbone_l,
            camera_neck=_neck_l,
            camera_head=_head_l,
            uncoupled_sonar_head=_head_l,
            fused_sonar_head=_head_l,
            fusion=dict(
                type="CrossAttnFusion",
                cam_channels=[256, 256, 256],
                sonar_channels=[256, 256, 256],
                embed_dim=256,
                num_heads=8,
                layer_scale_init=0.0,
                pos_embed=True,
            ),
            loss_weights=dict(camera=1.0, fused_sonar=0.5,
                              uncoupled_sonar=0.5),
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
                score_thr=0.01,
                nms=dict(type="nms", iou_threshold=0.65),
                max_per_img=100,
            ),
        ),
        teacher_checkpoint="work_dirs/e10_t_b2_yoloxl_teacher_seed0/"
                           "epoch_100.pth",
        teacher_input="clear",
        obj_weight=1.0,
        score_weight=1.0,
        tau=2.0,
        bg_floor=0.01,
        routes={"camera": "camera", "sonar": "sonar",
                "sonar_uncoupled": "sonar"},
    ),
)
