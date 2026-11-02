"""E11 D-self arm: self-distillation control -- the student's own recipe
as its own teacher (B3's epoch_100 checkpoint: same S capacity, same
degradation-augmented training), queried on the degraded view.

D-self is the no-capacity / no-privilege cell of the F2 §6.3 2x2: it
prices plain online KD from an equally-strong, equally-degraded model,
so capacity = (M1 - D-self) + (M3 - M2) and privilege = (M2 - D-self) +
(M3 - M1) are measured against the right floor.  KD settings equal
M3's (tau=2, obj+score, same routes, no reach / feature KD).  The
embedded teacher dict must equal e10_b3_degaug_yoloxs's model (B3);
tests/test_f2_configs.py asserts the two match.
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
        teacher_checkpoint="work_dirs/e10_b3_degaug_yoloxs_seed0/"
                           "epoch_100.pth",
        teacher_input="degraded",
        obj_weight=1.0,
        score_weight=1.0,
        tau=2.0,
        bg_floor=0.01,
        routes={"camera": "camera", "sonar": "sonar",
                "sonar_uncoupled": "sonar"},
    ),
)
