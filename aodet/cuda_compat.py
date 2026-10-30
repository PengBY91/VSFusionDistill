"""Runtime CUDA-compat shims for GPU environments with CPU-only mmcv.

The GPU training environment (``.venv-gpu``, see env/ENVIRONMENT.md) uses
torch 2.7.1+cu128 -- the first line with native Blackwell (sm_120, RTX
5090) kernels -- because the pinned cu121 stack cannot run there at all
("no kernel image available": cu121 wheels ship sm_50..sm_90 cubins, no
PTX fallback).  OpenMMLab publishes no mmcv wheels past cu121/torch2.4,
and mmdet 3.3.0 requires mmcv<2.2.0, so the GPU env builds mmcv 2.1.0
from source CPU-only (no nvcc needed).

That split is sound for this codebase: the training path never touches a
compiled mmcv op (YOLOX losses/SimOTA are pure torch, our paired
transforms use mmcv.transforms only).  The one compiled-op dependency is
inference-time NMS: ``mmcv.ops.nms`` dispatches by tensor device, so a
CPU-only ext raises on CUDA boxes.  ``install_nms_cpu_fallback()`` wraps
the YOLOX-head NMS to run on CPU and move results back -- at our scale
(single-target sonar fan at 640x640) that costs milliseconds per image.
It self-detects and stays a no-op where CUDA NMS already works, so the
same entry points run unchanged on the CPU/debug env and on a full-GPU
mmcv build.
"""

import torch


def _gpu_nms_available():
    """Probe whether mmcv's compiled NMS runs on CUDA tensors."""
    from mmcv.ops import nms

    if not torch.cuda.is_available():
        return True  # CPU env: CPU ext handles everything
    try:
        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]], device="cuda")
        scores = torch.tensor([0.9], device="cuda")
        nms(boxes, scores, 0.5)
        return True
    except Exception:
        return False


def install_nms_cpu_fallback():
    """Patch YOLOX-head batched_nms to CPU when mmcv lacks CUDA NMS."""
    if _gpu_nms_available():
        return False

    import mmdet.models.dense_heads.yolox_head as yh

    orig = yh.batched_nms
    if getattr(orig, "_ao_cpu_fallback", False):
        return True  # already installed

    def cpu_batched_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
        dev = boxes.device
        dets, keep = orig(boxes.cpu(), scores.cpu(), idxs.cpu(),
                          nms_cfg, class_agnostic)
        return dets.to(dev), keep.to(dev)

    cpu_batched_nms._ao_cpu_fallback = True
    yh.batched_nms = cpu_batched_nms
    return True
