"""E11-3: B3 degradation-augmented fusion (D0-D4, p=0.2 each, no
D-inf) on the full balanced UXO split (seeds 0/1/2).

Derived from e10_b3_degaug_yoloxs.py; per the E11 pre-registration
only the data and the schedule length change (same delta as
e11_b0/e11_b2).

Data: data/paired/uxo_full -- 47 recordings kept by the frozen filter,
split manifest sha256 ecfe1325..., train 27 recordings / 1650 frames;
sonar GT written under the certified calibrations (--calib-from basis,
doc/logs/2026-11-03-e11-final-gt-calib-drift.md section 7).

Schedule: 3x the E10 same-arm iterations = 3 x 100 epochs x 210
iters (1680 sample-train frames, batch 8) = 63,000 iters; the full
split trains on 1650 frames -> 207 iters/epoch -> 304 epochs = 62,928
iters (2.996x).  Warmup (5) and constant tail (15) keep their absolute
epoch values, matching e11_b0/e11_b2.

E11-3 launch gate (B-class amendment 1): runs only when E11-2 has been
accounted and either judged mutual poisoning (B3 recovery readout) or
route A/B continues (B3 is the baseline of the later M series); route A
was set by E11-1 criterion 2.  Evaluation reads val only (test stays
sealed); every tools/test.py invocation passes an explicit --split.
"""

_base_ = "e10_b3_degaug_yoloxs.py"

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
