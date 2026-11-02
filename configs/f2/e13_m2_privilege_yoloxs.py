"""E13 M2 arm: privilege-only distillation -- a SAME-capacity (S)
clear-view fusion teacher into the degradation-augmented student
(F2 §2.2 M2, the privilege cell of the 2x2).

Teacher = e13_t_b2s_nominal_yoloxs (the T-L recipe at S: nominal D0 +
sonar scale jitter), always queried on the clear camera
(teacher_input='clear').  M2 - D-self isolates privilege at equal
capacity; M3 - M2 isolates capacity at equal privilege (F2 §6.3).  KD
settings equal M3's (tau=2, obj+score, same routes, no reach / feature
KD).  The embedded teacher dict must equal e13_t_b2s_nominal_yoloxs's
model; tests/test_f2_configs.py asserts the two match.
"""

_base_ = ["e10_b3_degaug_yoloxs.py"]

_yolox_s_ckpt = ("https://download.openmmlab.com/mmdetection/v2.0/yolox/"
                 "yolox_s_8x8_300e_coco/"
                 "yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth")

_backbone_s = dict(
    type="CSPDarknet",
    deepen_factor=0.33,
    widen_factor=0.5,
    out_indices=(2, 3, 4),
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_s_ckpt,
                  prefix="backbone."),
)
_neck_s = dict(
    type="YOLOXPAFPN",
    in_channels=[128, 256, 512],
    out_channels=128,
    num_csp_blocks=1,
    init_cfg=dict(type="Pretrained", checkpoint=_yolox_s_ckpt,
                  prefix="neck."),
)
_head_s = dict(
    type="YOLOXHead",
    num_classes=3,
    in_channels=128,
    feat_channels=128,
)

model = dict(
    distillation=dict(
        teacher=dict(
            type="TwoStreamYOLOX",
            data_preprocessor=dict(
                type="PairedDataPreprocessor",
                pad_size_divisor=32,
            ),
            sonar_backbone=_backbone_s,
            sonar_neck=_neck_s,
            camera_backbone=_backbone_s,
            camera_neck=_neck_s,
            camera_head=_head_s,
            uncoupled_sonar_head=_head_s,
            fused_sonar_head=_head_s,
            fusion=dict(
                type="CrossAttnFusion",
                cam_channels=[128, 128, 128],
                sonar_channels=[128, 128, 128],
                embed_dim=128,
                num_heads=4,
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
        teacher_checkpoint="work_dirs/e13_t_b2s_nominal_yoloxs_seed0/"
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
