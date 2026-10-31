"""Tests for the GT validity gate tool (tools/check_gt_validity.py)."""

import numpy as np

from tools.check_gt_validity import measure


def _img_with_blob(h=780, w=128, row=300, peak=180, bg=20):
    img = np.full((h, w), bg, np.uint8)
    img[row - 4:row + 4, 50:70] = peak
    return img


def test_measure_dy_and_contrast_on_aligned_box():
    m = measure(_img_with_blob(), (50, 295, 20, 10))
    dy, contrast, peak = m
    assert dy <= 2.0                      # box centred on the blob
    assert contrast > 100                 # bright blob vs dim ring
    assert peak > 40


def test_measure_detects_row_offset():
    m = measure(_img_with_blob(), (50, 265, 20, 10))   # 35 px above blob
    dy = m[0]
    assert 27 <= dy <= 31
    assert m[1] == 0.0                     # box sits on background


def test_measure_handles_boundary_boxes():
    # box clipped at the image edge must not crash
    m = measure(_img_with_blob(), (118, 0, 20, 10))
    assert m is not None and m[0] >= 0


def test_measure_empty_overlap_returns_none():
    assert measure(_img_with_blob(), (130, 0, 10, 10)) is None
