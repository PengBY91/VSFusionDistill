"""Paired UXO dataset + dataloaders (setup doc §4.4(1),(3)).

data_root points at a converted paired-format root (see
aodet.datasets.converters.uxo_to_paired).  Train pipeline order is fixed
by §4.4(3): geometry first, then the teacher/student fork, then pack.
No Mosaic/MixUp anywhere (§4.4(2)).  Sync flips default off; only turn
them on after the sonar-format precondition (0°-centred bearing axis)
is re-verified for this data.
"""

dataset_type = "PairedSonarCameraDataset"
data_root = "data/paired/uxo/"

img_scale = (640, 640)

train_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=img_scale, sonar_size=img_scale, keep_ratio=False),
    dict(type="AOPairedSyncFlip", prob=0.0, allow=True),
    dict(type="AOSampleDegLevel"),
    # dinf_mode="zero": training batches stay dense — a per-sample D∞ draw
    # becomes a black camera frame (no light reaches the sensor), never a
    # hole in the batch tensor.  "drop" would kill the camera stream for
    # every mixed batch (PairedDataPreprocessor drops per batch).
    dict(type="AOStudentCameraView", dinf_mode="zero", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

test_pipeline = [
    dict(type="AOLoadPaired"),
    dict(type="AOPairedResize",
         camera_size=img_scale, sonar_size=img_scale, keep_ratio=False),
    # degradation level comes from the dataset's fixed deg_level field
    # (tools/test.py overrides it per level of the D0-Dinf scan);
    # batch_size=1 keeps the D∞ drop per-sample honest (camera absent)
    dict(type="AOStudentCameraView", dinf_mode="drop", jitter=0.08),
    dict(type="AOPackPairedInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="train",
        sonar_zero_centred=True,
        pipeline=train_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        split="test",
        sonar_zero_centred=True,
        pipeline=test_pipeline,
    ),
)
