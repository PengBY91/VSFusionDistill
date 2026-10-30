"""Fixed training recipe shared by all F2 methods (fairness constraint).

YOLOX-style SGD + cosine with warmup.  lr is written for batch_size=8
(single 24GB GPU); scale linearly if the batch changes.  No in-loop
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

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=5e-4),
    clip_grad=dict(max_norm=35, norm_type=2),
)

param_scheduler = [
    # warmup is already iter-based (by_epoch=False) -- no convert flag
    # (mmengine asserts convert_to_iter_based only on epoch-based entries)
    dict(type="LinearLR",
         start_factor=1e-4,
         by_epoch=False,
         begin=0,
         end=500),
    dict(type="CosineAnnealingLR",
         eta_min=5e-4,
         by_epoch=True,
         begin=1,
         end=max_epochs,
         convert_to_iter_based=True),
]

auto_scale_lr = dict(enable=False, base_batch_size=8)
