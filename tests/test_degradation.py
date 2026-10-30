"""Tests for the camera degradation operator (setup doc §4.5, row 1).

Assertions required by the plan:
  * higher level -> contrast and mean direct transmission strictly decrease;
  * the cast-only control leaves brightness and structure unchanged;
  * the same (sample_id, level, seed) reproduces pixel-identical output;
  * dinf outputs an all-zero frame.
"""

import numpy as np
import pytest

from aodet.datasets.transforms.camera_degradation import (
    ALL_LEVELS,
    LEVELS,
    DegradationParams,
    color_cast,
    degrade_camera,
    sample_params,
)

RNG = np.random.default_rng(20261030)


def _synthetic_scene(h: int = 240, w: int = 320) -> np.ndarray:
    """A scene with objects on a textured background: contrast measurable."""
    img = RNG.uniform(0.25, 0.45, size=(h, w, 3))
    for _ in range(12):
        y0, x0 = RNG.integers(0, h - 40), RNG.integers(0, w - 40)
        hh, ww = RNG.integers(18, 40), RNG.integers(18, 40)
        img[y0:y0 + hh, x0:x0 + ww] = RNG.uniform(0.55, 0.95, size=(hh, ww, 3))
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _gray(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float64) @ np.array([0.299, 0.587, 0.114])


@pytest.fixture(scope="module")
def scene() -> np.ndarray:
    return _synthetic_scene()


def test_levels_exposed():
    assert LEVELS == ("d0", "d1", "d2", "d3", "d4", "dinf")
    assert "cast" in ALL_LEVELS


def test_d0_is_identity(scene):
    out = degrade_camera(scene, sample_id="s1", level="d0", seed=7)
    np.testing.assert_array_equal(out, scene)


def test_dinf_is_black(scene):
    out = degrade_camera(scene, sample_id="s1", level="dinf", seed=7)
    assert out.dtype == scene.dtype
    assert out.shape == scene.shape
    assert not out.any()


def test_direct_transmission_monotonic(scene):
    """Mean direct transmission strictly decreases d0 > d1 > ... > d4."""
    ts = [
        sample_params("mono-sample", lvl, seed=3).mean_direct_transmission
        for lvl in ("d0", "d1", "d2", "d3", "d4")
    ]
    assert ts[0] == 1.0
    assert all(a > b for a, b in zip(ts, ts[1:])), ts


def test_contrast_monotonic(scene):
    """Gray RMS contrast strictly decreases d0 > d1 > ... > d4."""
    stds = []
    for lvl in ("d0", "d1", "d2", "d3", "d4"):
        out = degrade_camera(scene, sample_id="mono-sample", level=lvl, seed=3)
        stds.append(_gray(out).std())
    assert all(a > b for a, b in zip(stds, stds[1:])), stds


def test_monotonic_across_samples_and_seeds(scene):
    """Monotonicity must survive per-sample jitter: check 8 random ids."""
    ids = [f"frame-{i:04d}" for i in range(8)]
    for sid in ids:
        for seed in (0, 1):
            ts = [
                sample_params(sid, lvl, seed=seed).mean_direct_transmission
                for lvl in ("d1", "d2", "d3", "d4")
            ]
            assert all(a > b for a, b in zip(ts, ts[1:])), (sid, seed, ts)


def test_wavelength_ordering(scene):
    """Under water red dies first: t_R < t_G < t_B at every level."""
    for lvl in ("d1", "d2", "d3", "d4"):
        t = sample_params("s1", lvl, seed=0).direct_transmission
        assert t[0] < t[1] < t[2], (lvl, t)


def test_determinism(scene):
    """Same (sample_id, level, seed) -> pixel-identical output."""
    for lvl in ("d1", "d2", "d4", "cast", "dinf"):
        a = degrade_camera(scene, sample_id="det-1", level=lvl, seed=5)
        b = degrade_camera(scene, sample_id="det-1", level=lvl, seed=5)
        np.testing.assert_array_equal(a, b)
    # different seed or id must (almost surely) differ at d1..d4
    a = degrade_camera(scene, sample_id="det-1", level="d3", seed=5)
    b = degrade_camera(scene, sample_id="det-2", level="d3", seed=5)
    c = degrade_camera(scene, sample_id="det-1", level="d3", seed=6)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_params_deterministic_and_versioned():
    p1 = sample_params("xyz", "d2", seed=11)
    p2 = sample_params("xyz", "d2", seed=11)
    assert p1 == p2
    assert isinstance(p1, DegradationParams)
    assert 0.0 < p1.mean_direct_transmission < 1.0


def test_cast_preserves_luma_and_structure(scene):
    """Cast control: gray mean/std/pixels unchanged (quantization aside)."""
    out = color_cast(scene, sample_id="cast-1", seed=9)
    g0, g1 = _gray(scene), _gray(out)
    # uint8 rounding moves each pixel's gray by ≤0.5; the systematic part
    # of that quantization can shift the std by the same order — thresholds
    # below bound exactly that, nothing more
    assert abs(g0.mean() - g1.mean()) < 0.75
    assert abs(g0.std() - g1.std()) < 0.75
    np.testing.assert_allclose(g0, g1, atol=1.0)
    # but colour *is* shifted: R channel darkens, B brightens on average
    assert out[..., 0].mean() < scene[..., 0].mean()
    assert out[..., 2].mean() > scene[..., 2].mean()


def test_cast_preserves_luma_exactly_in_float(scene):
    """Pre-quantization (float dtype) luma preservation is exact."""
    f = scene.astype(np.float64) / 255.0
    out = color_cast(f, sample_id="cast-1", seed=9)
    y0 = f @ np.array([0.299, 0.587, 0.114])
    y1 = out @ np.array([0.299, 0.587, 0.114])
    np.testing.assert_allclose(y0, y1, atol=1e-12)


def test_dtype_and_shape_preserved(scene):
    out = degrade_camera(scene, sample_id="s", level="d4", seed=0)
    assert out.dtype == np.uint8 and out.shape == scene.shape
    f = scene.astype(np.float64) / 255.0
    outf = degrade_camera(f, sample_id="s", level="d4", seed=0)
    assert outf.dtype == np.float64


def test_rejects_bad_inputs(scene):
    with pytest.raises(ValueError):
        degrade_camera(scene, sample_id="s", level="d9")
    with pytest.raises(ValueError):
        degrade_camera(scene[..., 0], sample_id="s", level="d1")  # not HxWx3


def test_stream_of_levels_never_crashes(scene):
    for lvl in ALL_LEVELS:
        out = degrade_camera(scene, sample_id="sweep", level=lvl, seed=0)
        assert out.shape == scene.shape
