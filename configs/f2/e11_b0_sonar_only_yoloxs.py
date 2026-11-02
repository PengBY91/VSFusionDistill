"""E11-1: B0 sonar-only on the full balanced UXO split (seeds 0/1/2).

Derived from e09_s0_b0_jitter_yoloxs.py (same recipe + fixed sonar
scale jitter); per the E11 pre-registration only the data and the
schedule length change.

Data: data/paired/uxo_full -- 47 recordings kept by the frozen filter
(blob frames >= 30, trimmed row-residual p50 <= 20 px, az rms <= 3 deg on
the endpoint-trimmed window; trim = 20% off both ends of the capped
frame order), split manifest sha256 ecfe1325..., train 27 recordings /
1650 frames.

Schedule: length fixed at 3x the E09/E10 same-arm iterations.
E09b ran 100 epochs x 210 iters/epoch (1680 train frames, batch 8) =
21,000 iters; 3x = 63,000.  The full split trains on 1650 frames ->
207 iters/epoch, so max_epochs = 304 gives 62,928 iters (2.996x).
Warmup (5) and constant tail (15) keep their absolute epoch values,
which keeps their absolute iteration counts on par with E09b's.

Evaluation reads val only (the test split stays sealed until test
time); every tools/test.py invocation passes an explicit --split val
(or val_id for the health check).  The test_dataloader below defaults
to val so a flag-less invocation can never read test.
"""

_base_ = "e09_s0_b0_jitter_yoloxs.py"

data_root = "data/paired/uxo_full/"

train_dataloader = dict(dataset=dict(data_root=data_root))

test_dataloader = dict(dataset=dict(data_root=data_root, split="val"))

max_epochs = 304
base_lr = 0.01 * 8 / 64

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs,
                 val_interval=10)

param_scheduler = [
    dict(type="QuadraticWarmupLR",
         by_epoch=True,
         begin=0,
         end=5,
         convert_to_iter_based=True),
    dict(type="CosineAnnealingLR",
         eta_min=base_lr * 0.05,
         begin=5,
         T_max=max_epochs - 15,
         end=max_epochs - 15,
         by_epoch=True,
         convert_to_iter_based=True),
    dict(type="ConstantLR",
         by_epoch=True,
         factor=1,
         begin=max_epochs - 15,
         end=max_epochs),
]
