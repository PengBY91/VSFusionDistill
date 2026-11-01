"""E10 teacher: YOLOX-L two-stream fusion, nominal (D0) clear camera.

F2 §2.2 M3's "big fusion teacher that sees the clear view".  Trained
exactly like B2 (nominal D0 only) but at YOLOX-L capacity, both streams
initialised from COCO-pretrained YOLOX-L (the E01 gate validated the S
sibling of this checkpoint family).  The sonar side keeps the E09b scale
jitter so the teacher's sonar heads are not scale-locked to the train
domain (the teacher must be *strong*, E10 pre-registration gate 1).

Its checkpoint feeds e10_m3_privileged_yoloxs via
``distillation.teacher_checkpoint`` (no random teachers -- review
2026-11-01 #1).  Keep the L block definitions in sync with that config;
tests/test_f2_configs.py asserts the two model dicts match.
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
    sonar_backbone=_backbone_l,
    sonar_neck=_neck_l,
    camera_backbone=_backbone_l,
    camera_neck=_neck_l,
    camera_head=_head_l,
    uncoupled_sonar_head=_head_l,
    fused_sonar_head=_head_l,
    fusion=dict(
        cam_channels=[256, 256, 256],
        sonar_channels=[256, 256, 256],
        embed_dim=256,
        num_heads=8,
        layer_scale_init=0.0,
        pos_embed=True,
    ),
)

# AMP (fp16, dynamic loss scale): two-stream YOLOX-L at batch 8 in fp32
# needs ~26-30 GB activations and does not fit 24 GB -- the 2026-11-01
# 08:48 fp32 run sat at 97.6% of the card in an allocator retry loop
# (~90 s/iter, weeks to finish), and PYTORCH_CUDA_ALLOC_CONF=
# expandable_segments (the no-numerics-change fix) dies on first
# allocation under WSL2 ("CUDA driver error: out of memory", VMM
# unsupported).  AMP halves activation memory (fits with headroom) and
# is the mmdet YOLOX reference recipe.  Deviation from the fp32 recipe
# of every other arm: documented here + doc/experiments.md E10; the
# teacher is quality-gated before use, so the deviation is bounded.
optim_wrapper = dict(type="AmpOptimWrapper", loss_scale="dynamic")
