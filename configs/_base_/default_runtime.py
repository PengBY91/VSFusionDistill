default_scope = "mmdet"

# registers PairedSonarCameraDataset, AO* transforms, TwoStreamYOLOX,
# PairedDataPreprocessor (aodet never edits site-packages)
custom_imports = dict(
    imports=["aodet.datasets", "aodet.models"], allow_failed_imports=False)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=10, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer")
