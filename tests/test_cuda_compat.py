"""Tests for the CPU-only-mmcv NMS fallback shim (aodet.cuda_compat)."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aodet.cuda_compat as cc


def test_noop_when_nms_works(monkeypatch):
    """When mmcv NMS handles the active device, install() must be a
    no-op (returns False, leaves batched_nms untouched).  Forced via
    monkeypatch: on the GPU env the real probe is False (CPU-only mmcv
    ops), so the unforced version would be env-dependent and would
    leak an installed patch into the rest of the suite."""
    import mmdet.models.dense_heads.yolox_head as yh

    monkeypatch.setattr(cc, "_gpu_nms_available", lambda: True)
    before = yh.batched_nms
    assert cc.install_nms_cpu_fallback() is False
    assert yh.batched_nms is before


def test_fallback_wraps_and_preserves_results(monkeypatch):
    """Forced-fallback path: CPU NMS inside, results returned on the
    input device, identical keep set to a direct CPU call, idempotent."""
    import mmdet.models.dense_heads.yolox_head as yh
    from mmcv.ops import batched_nms

    orig = yh.batched_nms
    monkeypatch.setattr(cc, "_gpu_nms_available", lambda: False)
    try:
        assert cc.install_nms_cpu_fallback() is True
        assert cc.install_nms_cpu_fallback() is True  # idempotent

        boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0],
                              [1.0, 1.0, 11.0, 11.0],
                              [50.0, 50.0, 60.0, 60.0]])
        scores = torch.tensor([0.9, 0.8, 0.7])
        idxs = torch.tensor([0, 0, 1])
        cfg = dict(type="nms", iou_threshold=0.5)

        dets_ref, keep_ref = orig(boxes, scores, idxs, cfg)
        dets, keep = yh.batched_nms(boxes, scores, idxs, cfg)
        assert keep.tolist() == keep_ref.tolist()
        assert torch.allclose(dets, dets_ref)
    finally:
        yh.batched_nms = orig
