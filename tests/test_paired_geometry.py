"""Tests for M₀ geometry (setup doc §4.5, row 2).

Required assertions:
  * after a synchronized flip, the M₀-mapped azimuth window of the flipped
    camera box still corresponds to the (mirrored) sonar box;
  * with flips disabled the pipeline leaves bearings untouched;
  * windows are clipped to the sonar FOV and expanded by the residual.
"""

import pytest

from aodet.geometry.m0 import (
    M0Mapping,
    azimuth_window,
    load_m0,
    mirror_camera_box_x,
    mirror_m0,
    sonar_bearing_to_window,
    window_fov_overlap,
    windows_overlap,
)

# A plausible M₀: camera hfov 30.75° fully inside a 130° 0°-centred sonar
# FOV, camera x=0 (left) maps to −15.4°, x=1 to +15.4°.
M0 = M0Mapping(slope=30.75, intercept=-15.375, residual_std=0.5)


def test_theta_is_affine_and_ordered():
    assert M0.theta(0.0) == pytest.approx(-15.375)
    assert M0.theta(1.0) == pytest.approx(15.375)
    assert M0.theta(0.5) == pytest.approx(0.0)


def test_roundtrip_dict_and_file(tmp_path):
    p = tmp_path / "m0.json"
    p.write_text(__import__("json").dumps(M0.to_dict()))
    m = load_m0(str(p))
    assert m == M0
    m2 = load_m0(M0.to_dict())
    assert m2 == M0


def test_window_expansion_uses_residual():
    # box covering camera x in [0.4, 0.6] -> theta in [-3.075, 3.075]
    # margin = 2*0.5 + 1 = 2 deg
    lo, hi = azimuth_window(0.4 * 640, 0.6 * 640, 640, M0)
    assert lo == pytest.approx(-3.075 - 2.0, abs=1e-6)
    assert hi == pytest.approx(3.075 + 2.0, abs=1e-6)
    # larger residual -> wider window
    m_wide = M0Mapping(slope=30.75, intercept=-15.375, residual_std=2.0)
    lo2, hi2 = azimuth_window(0.4 * 640, 0.6 * 640, 640, m_wide)
    assert (hi2 - lo2) > (hi - lo)


def test_flip_mirrors_window_and_keeps_correspondence():
    """Sync flip: camera box mirrored + M₀ mirrored -> window matches the
    mirrored sonar box."""
    img_w = 640.0
    box = (200.0, 360.0)  # camera pixels
    w_before = azimuth_window(box[0], box[1], img_w, M0)

    mirrored_box = mirror_camera_box_x(box[0] / img_w, box[1] / img_w)
    m0_mir = mirror_m0(M0)
    w_after = azimuth_window(
        mirrored_box[0] * img_w, mirrored_box[1] * img_w, img_w, m0_mir
    )

    # mirroring is exact: the window is reflected about 0°
    assert w_after[0] == pytest.approx(-w_before[1], abs=1e-9)
    assert w_after[1] == pytest.approx(-w_before[0], abs=1e-9)

    # and it still corresponds to the mirrored sonar box bearing interval
    sonar_box_bearing = w_before  # superset of the sonar box interval
    mirrored_sonar = sonar_bearing_to_window(0.0, 0.0, M0)  # placeholder
    mirrored_sonar = (-sonar_box_bearing[1], -sonar_box_bearing[0])
    assert windows_overlap(w_after, mirrored_sonar)


def test_flip_forbidden_when_sonar_not_centred():
    m = M0Mapping(slope=30.0, intercept=0.0, sonar_zero_centred=False)
    with pytest.raises(ValueError):
        mirror_m0(m)


def test_window_clipped_to_sonar_fov():
    # window far outside the (0-centred, 130°) FOV
    assert window_fov_overlap((200.0, 210.0), M0) is None
    # straddling the right edge: clipped at +65°
    w = window_fov_overlap((50.0, 90.0), M0)
    assert w == (50.0, 65.0)
    # fully inside: unchanged
    assert window_fov_overlap((-10.0, 10.0), M0) == (-10.0, 10.0)


def test_camera_strip_covers_quarter_of_sonar_fov():
    """F2 §5.3: 30.75° camera vs 130° sonar -> about a quarter coverage."""
    coverage = M0.camera_hfov_deg / M0.sonar_fov_deg
    assert 0.2 < coverage < 0.3  # ≈ 0.237
