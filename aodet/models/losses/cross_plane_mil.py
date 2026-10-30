"""Cross-plane azimuth-window MIL distillation (F2 §5.3, L_xp).

The teacher's camera detections live in a front-view image plane; the
student's sonar head lives on a range–azimuth grid.  A monocular camera
provides no range, so the only geometrically honest correspondence is
"camera column -> sonar azimuth window, range unknown" (张力 4).  The loss
is therefore multiple-instance: for a teacher box ``b_c`` with objectness
``p^T_obj(b_c)``, mapped through M₀ to an azimuth window ``W(b_c)``::

    L_xp = Σ_{b_c} BCE( max_{(r,θ) ∈ W(b_c)} p_s^obj(r, θ),  p^T_obj(b_c) )

Only **positive evidence** is transferred: the camera's vertical FOV does
not cover the sonar's, so a camera non-detection says nothing about the
sonar strip (PRISM's one-sided information principle).  Windows falling
(possibly partially) outside the sonar FOV are clipped; fully outside
windows are skipped (about 3/4 of bearings are outside a 30.75° camera FOV
inside a 130° sonar FOV).

The student objectness is read from one feature level of the sonar head
(default: level 0 = highest resolution, stride 8).  The sonar image is the
rectangular polar map (x = bearing, ascending; y = range) fixed by the
paired-format convention; the bearing axis of the feature grid is x.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from aodet.geometry.m0 import M0Mapping, azimuth_window, window_fov_overlap

__all__ = ["CrossPlaneMILLoss"]


class CrossPlaneMILLoss(torch.nn.Module):
    """L_xp between teacher camera boxes and student sonar objectness maps.

    Args:
        m0: the nominal mapping (loaded from F1's fit, never re-fitted).
        bearing_fov_deg: total sonar bearing FOV spanned by the sonar
            image's bearing axis (e.g. 130 for Oculus-like, 30 for ARIS-like
            after cropping — read from the paired dataset metadata).
        k_margin / base_margin_deg: azimuth-window expansion beyond the
            M₀ residual (F2 §5.3: expansion follows F1 L1's calibration
            tolerance; 2σ until that curve exists).
        level: which multi-level objectness map to supervise (default 0).
        bearing_axis: 'x' per the paired-format convention ('y' kept for
            the fan-image option deferred by TODO 14.3).
    """

    def __init__(
        self,
        m0: M0Mapping,
        bearing_fov_deg: float,
        k_margin: float = 2.0,
        base_margin_deg: float = 1.0,
        level: int = 0,
        bearing_axis: str = "x",
    ):
        super().__init__()
        self.m0 = m0
        self.bearing_fov_deg = float(bearing_fov_deg)
        self.k_margin = k_margin
        self.base_margin_deg = base_margin_deg
        self.level = level
        if bearing_axis != "x":
            raise NotImplementedError(
                "fan-shaped sonar images (bearing on rays) are the deferred "
                "W2 decision; use the rectangular polar format")

    def forward(
        self,
        teacher_boxes: Sequence[Sequence[Tuple[float, float, float, float]]],
        teacher_obj: Sequence[Tensor],
        student_obj_logits: Sequence[Tensor],
        camera_img_w: float,
    ) -> Tuple[Tensor, Tensor]:
        """Compute the loss.

        Args:
            teacher_boxes: per image, list of (x1, y1, x2, y2) camera boxes
                in pixels (clear-view teacher detections above threshold).
            teacher_obj: per image, tensor of teacher objectness
                probabilities for those boxes.
            student_obj_logits: per-level objectness logits B×1×H×W from the
                student's sonar head (sigmoid applied here).
            camera_img_w: width of the (augmented) camera image the teacher
                saw — needed because M₀ works on normalized camera x.

        Returns:
            (loss, n_used): mean BCE over usable boxes and the count
            (windows fully outside the sonar FOV are skipped and counted).
        """
        logits = student_obj_logits[self.level]
        probs = logits.sigmoid()  # B,1,H,W
        b, _, h, w = probs.shape
        theta_lo, theta_hi = -self.bearing_fov_deg / 2, self.bearing_fov_deg / 2
        # bearing -> feature column (rectangular polar: x ascending bearing)
        col = lambda th: (th - theta_lo) / (theta_hi - theta_lo) * (w - 1)

        losses: List[Tensor] = []
        n_used = 0
        n_skip = 0
        for i, (boxes, objs) in enumerate(zip(teacher_boxes, teacher_obj)):
            if len(boxes) == 0:
                continue
            for (x1, y1, x2, y2), p_t in zip(boxes, objs.tolist()):
                window = azimuth_window(x1, x2, camera_img_w, self.m0,
                                        k_margin=self.k_margin,
                                        base_margin_deg=self.base_margin_deg)
                clip = window_fov_overlap(window, self.m0)
                if clip is None:
                    n_skip += 1
                    continue
                c1 = max(int(col(clip[0])), 0)
                c2 = min(int(col(clip[1])) + 1, w)
                if c2 <= c1:
                    n_skip += 1
                    continue
                p_max = probs[i, 0, :, c1:c2].max()
                losses.append(F.binary_cross_entropy(
                    p_max.clamp(1e-6, 1 - 1e-6),
                    probs.new_tensor(float(p_t))))
                n_used += 1
        if n_used == 0:
            return probs.sum() * 0.0, probs.new_tensor(0.0)
        return torch.stack(losses).mean(), probs.new_tensor(float(n_used))
