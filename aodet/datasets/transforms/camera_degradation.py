"""Camera degradation operator: levels D0–D4, D∞ and the color-cast control.

Implements the four-effect degradation stack used by the Degradation-aware
Foundation Fusion paper (see the F2 draft §2.1): light attenuation,
wavelength-dependent absorption (beta_R > beta_G > beta_B), scattering
``I = J * t_s + A_s * (1 - t_s)`` and Gaussian blur.  The paper released no
code, so this is a from-scratch implementation of the *protocol*, not a
reproduction of their exact parameters.

Design notes (all deliberate, see tests/test_degradation.py):

* **No depth map is available** for RSFusion/UXO camera frames, so the direct
  transmission ``t_c = exp(-(c_light + beta_c) * range_m)`` uses one nominal
  target range per image instead of a per-pixel depth.  ``range_m`` is a
  parameter, not a measurement.
* **Monotonicity by construction**: contrast of the direct term is
  ``t_c * std(J_c)`` and the veiling term adds a (near-)constant, so gray
  contrast and mean direct transmission are strictly decreasing in the level
  as long as the level gaps exceed the jitter range.  The base table below
  leaves >2x jitter margins between consecutive levels.
* **Determinism**: every random draw comes from a generator seeded by
  ``(crc32(sample_id), level_index, seed)`` — never from global RNG state.
  zlib.crc32 is used because Python's ``hash()`` is salted per process.

Negative control ``cast``: per-channel gains normalized so that luma is
preserved exactly (grayscale image unchanged) — the paper reports fusion
gains of +0.0% under pure color shift, so a trained model must *not* treat
this level like real turbidity.

``dinf`` returns an all-zero image (``dinf_mode="zero"``).  "Drop the input
entirely" is a dataset-level decision (the camera tensor is not produced at
all), implemented by the paired dataset, not by this operator.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "LEVELS",
    "CONTROL_LEVELS",
    "ALL_LEVELS",
    "DegradationParams",
    "sample_params",
    "degrade_camera",
    "color_cast",
    "level_index",
    "AODegradCamera",
]

LEVELS = ("d0", "d1", "d2", "d3", "d4", "dinf")
CONTROL_LEVELS = ("cast",)
ALL_LEVELS = LEVELS + CONTROL_LEVELS

# Luma weights (Rec. 601), used for the cast-only normalization and the
# gray-contrast monotonicity tests.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)

# Base parameters per level. d0 is the identity. Columns:
#   c_light  : non-selective light attenuation coefficient [1/m]
#   beta_k   : wavelength absorption scale; per-channel beta = k * (R,G,B)
#   veil_a   : veiling-light scale; A_c = veil_a * (0.10, 0.28, 0.40)
#   sigma    : Gaussian blur sigma [px] at 640-px nominal input
# Median ranges seen in UXO/RSFusion-like basins are a few metres, hence
# range_m base = 3.0.
_BASE_RANGE_M = 3.0
_VEIL_COLOR = np.array([0.10, 0.28, 0.40], dtype=np.float64)  # B > G > R
_BETA_RATIO = np.array([4.0, 1.6, 0.8], dtype=np.float64)  # R > G > B
_BASE: Dict[str, Dict[str, float]] = {
    "d0": dict(c_light=0.0, beta_k=0.0, veil_a=0.0, sigma=0.0),
    "d1": dict(c_light=0.03, beta_k=0.015, veil_a=0.25, sigma=0.7),
    "d2": dict(c_light=0.07, beta_k=0.040, veil_a=0.45, sigma=1.4),
    "d3": dict(c_light=0.13, beta_k=0.080, veil_a=0.70, sigma=2.4),
    "d4": dict(c_light=0.22, beta_k=0.140, veil_a=1.00, sigma=3.8),
}
# Relative jitter applied to every physical parameter.  0.08 keeps the
# consecutive-level gaps in mean direct transmission non-overlapping
# (worst case d1/d2 margin is ~2.3x the jitter span).
DEFAULT_JITTER = 0.08


def level_index(level: str) -> int:
    """Stable integer id of a level, used for RNG seeding."""
    try:
        return ALL_LEVELS.index(level)
    except ValueError as exc:
        raise ValueError(
            f"unknown degradation level {level!r}; expected one of {ALL_LEVELS}"
        ) from exc


@dataclass(frozen=True)
class DegradationParams:
    """Sampled parameters for one (sample_id, level, seed) draw."""

    level: str
    range_m: float
    c_light: float
    beta_rgb: Tuple[float, float, float]
    veil_rgb: Tuple[float, float, float]
    sigma_blur: float

    @property
    def direct_transmission(self) -> Tuple[float, float, float]:
        """Per-channel direct transmission t_c = exp(-(c+beta_c)*range)."""
        c = np.exp(-(self.c_light + np.asarray(self.beta_rgb)) * self.range_m)
        return tuple(float(v) for v in c)

    @property
    def mean_direct_transmission(self) -> float:
        return float(np.mean(self.direct_transmission))


def _seeded_rng(sample_id: Any, level: str, seed: int) -> np.random.Generator:
    sid = str(sample_id).encode("utf-8")
    entropy = [zlib.crc32(sid), level_index(level), int(seed)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _jitter(rng: np.random.Generator, value: float, jitter: float) -> float:
    return float(value * rng.uniform(1.0 - jitter, 1.0 + jitter))


def sample_params(
    sample_id: Any,
    level: str,
    seed: int = 0,
    jitter: float = DEFAULT_JITTER,
) -> DegradationParams:
    """Deterministically sample the physical parameters for one draw."""
    if level not in _BASE and level != "dinf":
        raise ValueError(f"sample_params does not serve level {level!r}; "
                         f"use color_cast()/degrade_camera() for controls")
    if level in ("d0", "dinf"):
        # Identity / black-out: no free parameters, but keep the draw
        # deterministic anyway so callers cannot tell the difference.
        _seeded_rng(sample_id, level, seed)
        return DegradationParams(level, _BASE_RANGE_M, 0.0, (0.0, 0.0, 0.0),
                                 (0.0, 0.0, 0.0), 0.0)
    rng = _seeded_rng(sample_id, level, seed)
    base = _BASE[level]
    beta = tuple(
        float(v)
        for v in _jitter(rng, base["beta_k"], jitter) * _BETA_RATIO
    )
    veil = tuple(
        float(v)
        for v in np.clip(
            _jitter(rng, base["veil_a"], jitter) * _VEIL_COLOR, 0.0, 1.0
        )
    )
    return DegradationParams(
        level=level,
        range_m=_jitter(rng, _BASE_RANGE_M, jitter),
        c_light=_jitter(rng, base["c_light"], jitter),
        beta_rgb=beta,
        veil_rgb=veil,
        sigma_blur=_jitter(rng, base["sigma"], jitter),
    )


def _to_float(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got shape {img.shape}")
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float64), 0.0, 1.0)
    raise TypeError(f"unsupported dtype {img.dtype}")


def apply_params(img: np.ndarray, p: DegradationParams) -> np.ndarray:
    """Apply the physics of ``p`` to a float image in [0, 1]; returns float.

    I_c = J_c * t_c + A_c * (1 - t_c), followed by Gaussian blur.
    """
    J = _to_float(img)
    t = np.asarray(p.direct_transmission, dtype=np.float64)  # (3,)
    A = np.asarray(p.veil_rgb, dtype=np.float64)  # (3,)
    I = J * t + A * (1.0 - t)
    I = np.clip(I, 0.0, 1.0)
    if p.sigma_blur > 0.0:
        # uint8 round-trip keeps cv2 happy on any channel order; sigma is
        # small enough that quantization is negligible against the effect.
        I8 = (I * 255.0 + 0.5).astype(np.uint8)
        for _ in range(1):  # single pass, matches the protocol's "blur" item
            I8 = cv2.GaussianBlur(I8, ksize=(0, 0), sigmaX=p.sigma_blur)
        I = I8.astype(np.float64) / 255.0
    return I


def _from_float(img_f: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if dtype == np.uint8:
        return (img_f * 255.0 + 0.5).astype(np.uint8)
    return img_f.astype(dtype)


def color_cast(
    img: np.ndarray,
    sample_id: Any,
    seed: int = 0,
    jitter: float = DEFAULT_JITTER,
) -> np.ndarray:
    """Negative control: color shift with luma preserved exactly.

    Channel gains are drawn toward the water direction (R attenuated, B
    boosted), then a common per-pixel offset restores the luma-weighted sum
    to the original value.  Because that offset can push bright pixels out
    of gamut, the final output is a per-pixel convex combination
    ``out = J + λ·(target − J)`` with the largest λ that stays inside
    [0, 1] — and since luma is linear and both endpoints have the original
    luma, luma(out) = luma(J) *exactly* for every λ, clipping included.
    The cast is only weakened on the few extreme pixels; the grayscale image
    is unchanged up to uint8 rounding.
    """
    rng = _seeded_rng(sample_id, "cast", seed)
    gain = 1.0 + np.array(
        [
            -rng.uniform(0.15, 0.45) * (1.0 + jitter),
            rng.uniform(0.0, 0.10),
            rng.uniform(0.15, 0.45),
        ]
    )
    J = _to_float(img)
    shifted = J * gain[None, None, :]
    # restore luma: add the per-pixel luma deficit to all channels equally
    delta = (J * _LUMA[None, None, :]).sum(axis=2, keepdims=True) - (
        shifted * _LUMA[None, None, :]
    ).sum(axis=2, keepdims=True)
    target = shifted + delta
    # per-pixel λ (shared across channels!): the largest value keeping
    # out = J + λ(target−J) inside [0,1] on every channel
    diff = target - J
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.where(
            diff > 0, (1.0 - J) / diff,
            np.where(diff < 0, (0.0 - J) / diff, np.inf),
        )
    lam = np.clip(np.nanmin(cap, axis=2, keepdims=True), 0.0, 1.0)
    out = J + lam * diff
    return _from_float(out, img.dtype)


def degrade_camera(
    img: np.ndarray,
    sample_id: Any,
    level: str,
    seed: int = 0,
    jitter: float = DEFAULT_JITTER,
    dinf_mode: str = "zero",
) -> np.ndarray:
    """Degrade a clean camera frame to ``level``.

    Args:
        img: HxWx3 uint8 (or float [0,1]) camera image.
        sample_id: stable id of the pair; drives parameter sampling.
        level: one of ``ALL_LEVELS`` ("d0".."d4", "dinf", "cast").
        seed: extra entropy, e.g. the training epoch-free per-draw seed.
        dinf_mode: "zero" returns a black frame; "drop" is not an image
            operation and belongs to the paired dataset.

    Returns an array with the same shape/dtype as ``img``.
    """
    if level == "cast":
        return color_cast(img, sample_id=sample_id, seed=seed, jitter=jitter)
    if level == "dinf":
        if dinf_mode != "zero":
            raise ValueError(f"unknown dinf_mode {dinf_mode!r}")
        _seeded_rng(sample_id, level, seed)  # keep the draw consumed
        return np.zeros_like(img)
    p = sample_params(sample_id, level, seed=seed, jitter=jitter)
    if level == "d0":
        # Identity: still run the pipeline (t=1, no veil, no blur) so that
        # d0 frames pass through the same code path.
        return _from_float(apply_params(img, p), img.dtype)
    return _from_float(apply_params(img, p), img.dtype)


# ---------------------------------------------------------------------------
# mmengine transform wrapper (lazy import so core tests need no mm* stack)
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised only with mmengine installed
    from mmengine.registry import TRANSFORMS
    from mmcv.transforms import BaseTransform

    @TRANSFORMS.register_module()
    class AODegradCamera(BaseTransform):
        """Apply ``degrade_camera`` inside an mmengine data pipeline.

        Expected ``results`` keys (set by PairedSonarCameraDataset):
          ``img`` (camera half only), ``sample_id``, ``deg_level``, ``deg_seed``.
        The fork point mandated by the setup doc §4.4(3) — geometric
        augmentation first, then copy for the teacher, then degrade the
        student's copy — is orchestrated by the dataset; this transform is
        only the physics step.
        """

        def __init__(self, jitter: float = DEFAULT_JITTER):
            super().__init__()
            self.jitter = jitter

        def transform(self, results: Dict[str, Any]) -> Dict[str, Any]:
            level = results.get("deg_level", "d0")
            seed = results.get("deg_seed", 0)
            results["img"] = degrade_camera(
                results["img"],
                sample_id=results["sample_id"],
                level=level,
                seed=seed,
                jitter=self.jitter,
            )
            results["deg_params"] = sample_params(
                results["sample_id"], level, seed=seed, jitter=self.jitter
            ) if level in _BASE else None
            return results

except ImportError:  # mmengine not installed (e.g. pure-numpy unit tests)
    class AODegradCamera:  # type: ignore[no-redef]
        """Stub; install mmengine to use the pipeline wrapper."""

        def __init__(self, *a: Any, **k: Any):
            raise ImportError("mmengine is required for AODegradCamera")
