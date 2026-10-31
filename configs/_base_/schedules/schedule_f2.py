"""Fixed training recipe shared by all F2 methods (fairness constraint).

YOLOX-style SGD + cosine with warmup.  lr is the YOLOX reference scaled
to batch_size=8 (single 24GB GPU); scale linearly if the batch changes.  No in-loop
validation: per-branch evaluation happens offline via tools/test.py
against saved prediction files (§4.4(6)).
"""

max_epochs = 100

train_cfg = dict(
    type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=10)

# no in-loop validation: per-branch evaluation runs offline via
# tools/test.py against saved prediction files (§4.4(6)).  test_cfg /
# test_evaluator are placeholders so Runner.from_cfg accepts the config
# together with test_dataloader (all three must be set or all None).
val_cfg = None
test_cfg = dict(type="TestLoop")
test_evaluator = dict(type="DummyMetric")

# YOLOX recipe (mmdet yolox_s_8xb8-300e_coco): lr 0.01 at batch 64 ->
# linear scaling to batch 8; no decay on norm/bias; nesterov; quadratic
# warmup -> cosine -> constant tail; EMA (review 2026-10-31 §1.4).  The
# previous lr 0.01 at batch 8 was 8x the reference, and its iter-based
# warmup overlapped the cosine schedule.
base_lr = 0.01 * 8 / 64
num_last_epochs = 15
warmup_epochs = 5

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=base_lr, momentum=0.9, weight_decay=5e-4,
                   nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    dict(type="QuadraticWarmupLR",
         by_epoch=True,
         begin=0,
         end=warmup_epochs,
         convert_to_iter_based=True),
    dict(type="CosineAnnealingLR",
         eta_min=base_lr * 0.05,
         begin=warmup_epochs,
         T_max=max_epochs - num_last_epochs,
         end=max_epochs - num_last_epochs,
         by_epoch=True,
         convert_to_iter_based=True),
    dict(type="ConstantLR",
         by_epoch=True,
         factor=1,
         begin=max_epochs - num_last_epochs,
         end=max_epochs),
]

# YOLOXModeSwitchHook is deliberately absent: it reaches for
# model.bbox_head, which TwoStreamYOLOX does not have.
custom_hooks = [
    dict(type="EMAHook", ema_type="ExpMomentumEMA", momentum=0.0001,
         update_buffers=True, priority=49),
]

auto_scale_lr = dict(enable=False, base_batch_size=8)
