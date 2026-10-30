"""Nominal mapping M₀: camera horizontal angle -> sonar bearing (F2 §2.1).

F2 reuses the M₀ fit produced by F1 (F1 §7.6, fitted on the RSFusion
training set) instead of re-fitting it.  This module therefore only *loads*
and *applies* an M₀, never estimates one.

Model: a per-dataset affine map from normalized camera x (0..1, left→right)
to sonar bearing θ in degrees, ``θ = slope * x + intercept``, together with
the fit's residual standard deviation used to expand azimuth windows:

    W(b_c) = [θ_center − w/2 − margin, θ_center + w/2 + margin],
    margin = k_margin * residual_std + base_margin_deg

The cross-plane distillation loss L_xp only fires when the window lies
(possibly partially) inside the sonar field of view, and only positive
evidence is transferred (F2 §5.3) — camera non-detections say nothing about
the sonar strip because the vertical apertures differ.

Flip rule (setup doc §4.4(2)): a synchronized horizontal flip of the camera
image corresponds to mirroring the sonar bearing (θ -> −θ) *only if* the
sonar image is 0°-centred.  The mirrored affine map keeps the slope and
moves the intercept to ``−(slope + intercept)`` (see :func:`mirror_m0`).
If the sonar image is not 0°-centred, synchronized flips are forbidden and
the pipeline must leave both sides untouched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

__all__ = ["M0Mapping", "load_m0", "azimuth_window", "window_fov_overlap",
           "mirror_m0", "mirror_camera_box_x"]

# Default: FOV widths of the Oculus MD750d-style sonar vs a GoPro-ish camera
# on the RSFusion rig (F2 §2.5: 130° sonar vs 30.75° camera).
DEFAULT_SONAR_FOV_DEG = 130.0
DEFAULT_CAMERA_HFOV_DEG = 30.75


@dataclass(frozen=True)
class M0Mapping:
    slope: float                 # deg per unit normalized camera x
    intercept: float             # θ at camera x = 0, degrees
    residual_std: float = 0.5    # fit residual std, degrees
    camera_hfov_deg: float = DEFAULT_CAMERA_HFOV_DEG
    sonar_fov_deg: float = DEFAULT_SONAR_FOV_DEG
    sonar_zero_centred: bool = True

    def theta(self, x_norm: float) -> float:
        return self.slope * float(x_norm) + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "residual_std": self.residual_std,
            "camera_hfov_deg": self.camera_hfov_deg,
            "sonar_fov_deg": self.sonar_fov_deg,
            "sonar_zero_centred": self.sonar_zero_centred,
        }


def load_m0(path_or_dict: Any) -> M0Mapping:
    """Load an M₀ produced by F1 (json file path or already-parsed dict)."""
    if isinstance(path_or_dict, (str, bytes)):
        with open(path_or_dict, "r", encoding="utf-8") as f:
            d = json.load(f)
    else:
        d = dict(path_or_dict)
    return M0Mapping(
        slope=float(d["slope"]),
        intercept=float(d["intercept"]),
        residual_std=float(d.get("residual_std", 0.5)),
        camera_hfov_deg=float(d.get("camera_hfov_deg",
                                    DEFAULT_CAMERA_HFOV_DEG)),
        sonar_fov_deg=float(d.get("sonar_fov_deg", DEFAULT_SONAR_FOV_DEG)),
        sonar_zero_centred=bool(d.get("sonar_zero_centred", True)),
    )


def mirror_m0(m: M0Mapping) -> M0Mapping:
    """M₀ after a synchronized horizontal flip (θ -> −θ).

    Composition of the two flips: the camera coordinate becomes
    ``x' = 1 − x`` and the bearing becomes ``θ' = −θ``, so the mirrored map
    is ``θ'(x') = −θ(1 − x') = slope·x' − (slope + intercept)`` — the slope
    is unchanged and the intercept moves to ``−(slope + intercept)``.

    Only valid when the sonar image is 0°-centred; raises otherwise.
    """
    if not m.sonar_zero_centred:
        raise ValueError(
            "synchronized flip requires a 0°-centred sonar image; "
            "the pipeline must skip flips for this dataset"
        )
    return M0Mapping(
        slope=m.slope,
        intercept=-(m.slope + m.intercept),
        residual_std=m.residual_std,
        camera_hfov_deg=m.camera_hfov_deg,
        sonar_fov_deg=m.sonar_fov_deg,
        sonar_zero_centred=m.sonar_zero_centred,
    )


def mirror_camera_box_x(x1: float, x2: float) -> Tuple[float, float]:
    """Camera box x-interval (any pixel/normalized unit, left/right ordered)
    after a horizontal flip of the camera image."""
    return (1.0 - max(x1, x2), 1.0 - min(x1, x2))


def azimuth_window(
    box_x1: float,
    box_x2: float,
    img_w: float,
    m0: M0Mapping,
    k_margin: float = 2.0,
    base_margin_deg: float = 1.0,
) -> Tuple[float, float]:
    """Azimuth window W(b_c) of a camera box, expanded by the M₀ residual.

    Args:
        box_x1, box_x2: box left/right pixel coordinates in the camera image.
        img_w: camera image width in pixels.
        m0: the loaded mapping.
        k_margin: window expansion in units of residual_std (F2 §5.3 says the
            expansion follows F1 L1's calibration-tolerance curve; 2σ is the
            default until that curve exists).
        base_margin_deg: constant safety margin in degrees.

    Returns:
        (theta_lo, theta_lo <= theta_hi) in degrees.
    """
    x1n, x2n = min(box_x1, box_x2) / img_w, max(box_x1, box_x2) / img_w
    t1, t2 = m0.theta(x1n), m0.theta(x2n)
    lo, hi = min(t1, t2), max(t1, t2)
    margin = k_margin * m0.residual_std + base_margin_deg
    return (lo - margin, hi + margin)


def window_fov_overlap(
    window: Tuple[float, float], m0: M0Mapping
) -> Optional[Tuple[float, float]]:
    """Clip an azimuth window to the sonar FOV; None if fully outside.

    The sonar FOV is [−fov/2, +fov/2] for a 0°-centred image.  F2 §5.3: L_xp
    only applies where the window falls inside the sonar FOV — 130° sonar vs
    30.75° camera means the strip covers about a quarter of the bearings.
    """
    lo, hi = window
    if hi < lo:
        lo, hi = hi, lo
    if not m0.sonar_zero_centred:
        # FOV anchored at the left edge of the sonar image instead
        fov_lo, fov_hi = 0.0, m0.sonar_fov_deg
    else:
        fov_lo, fov_hi = -m0.sonar_fov_deg / 2.0, m0.sonar_fov_deg / 2.0
    clo, chi = max(lo, fov_lo), min(hi, fov_hi)
    if clo >= chi:
        return None
    return (clo, chi)


def sonar_bearing_to_window(bearing_deg: float, width_deg: float,
                            m0: M0Mapping) -> Tuple[float, float]:
    """Bearing interval of a sonar box (used to cross-check windows against
    sonar GT boxes in the geometry tests)."""
    return (bearing_deg - width_deg / 2, bearing_deg + width_deg / 2)


def windows_overlap(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return max(a[0], b[0]) <= min(a[1], b[1])
