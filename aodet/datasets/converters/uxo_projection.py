"""UXO sonar-box projection: per-recording odom-frame calibration.

The UXO dataset ships camera GT boxes only.  This module produces the
sonar-side boxes and the cross-modal correspondence ``pi`` (pairs.csv)
by projecting the calibrated target position into each sonar frame.

Model (all in the gantry ``/odom`` frame, translations only -- verified
against the sample recordings, see the W1-W3 log 2026-10-30):

* Camera and sonar hang rigidly off the ar3 bracket; their odom origins
  per frame come from gantry.csv + the static chain of
  ``calibration/transforms.yaml`` (translations there are trustworthy).
* The gantry csv carries NO orientation, so per-frame yaw must be
  modelled.  Three of the five sample recordings drive a ~120 deg arc
  around the target.  On the real arcs the camera keeps the target
  within ~2 deg of the image centre for the whole traverse (verified
  against the GT boxes): the head is continuously RE-AIMED at the
  target, it does not yaw with the rail tangent (a tangent-locked head
  would show a growing in-frame drift).  The ``aim`` convention models
  this: R_head(i) = R_align(unit(P - c_head(i))) @ R_off, with R_align
  rotating the bracket z onto the line of sight (Rodrigues, vectorised)
  and R_off the constant mis-aim.  The alternatives -- polar yaw
  (yaw_i = arc polar angle; the tangent-locked trolley) and the
  straight-traverse constant rotation -- remain candidates and the fit
  picks by cost.  Whether a recording is an arc or a straight traverse
  is decided by how much better a circle explains the path than a line
  (a Kasa fit through a noisy straight path returns a bogus finite
  circle).
* Per-recording unknowns, fitted jointly with scipy least_squares
  (Huber): free camera rotation ``R_cam0`` (absorbs the unknown
  pan/tilt convention), target position ``P``, sonar rotation
  ``R_son0`` (fit independently of the camera: the yaml chain's
  sonar-camera relative rotation does NOT hold across recordings --
  the head is re-aimed between runs), and a range bias ``b_rng``
  (front-face echo vs box centre, sound-speed residuals).

Observations: camera GT box-centre bearings (linear GoPro intrinsics --
the recordings are shot in GoPro "Linear" FOV, no undistortion needed)
and sonar blob echoes (bright compact returns), converted to (slant
range, beam angle) via ``aodet.geometry.aris_beam_table``.

Ground-truth policy: where a confident blob echo sits within the
consistency gate of the projection, the observed blob bbox (padded) is
the sonar box (``source="blob"``); otherwise the geometric projection
with class-length margins (``source="projected"``).  The calibration
report records |projected - observed| statistics per recording (the
"projection error" of F2 TODO 14.2 item 4).
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from ...geometry.aris_beam_table import (beam_center_deg, col_of_angle,
                                         range_of_row, row_of_range)

__all__ = ["CalibResult", "FrameObs", "calibrate", "load_observations",
           "project_uxo_sonar", "detect_blobs", "fit_circle", "predict"]

# --- static chain from calibration/transforms.yaml (quats x,y,z,w) --------
_T_PC_AR3 = ((0.0, 0.174, -0.338), (0.0, 1.0, 0.0, 0.0))
_T_ARIS_AR3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
_T_SONAR_ARIS = ((-0.108725, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5))
_T_HOLE_AR3 = ((0.1265, 0.023, -0.1055), (0.0, 0.0, 0.0, 1.0))
_T_AXIS_HOLE = ((0.02125, 0.0, -0.0075), (0.0, 0.0, 0.0, 1.0))
_T_CAM_AXIS = ((0.01925, -0.0195, -0.0415), (0.5, 0.5, 0.5, 0.5))

# GoPro "Linear" intrinsics (calibration/gopro_intrinsics_*.txt), 1920x1080
K_LINEAR = np.array([[1473.05, 0, 969.39],
                     [0, 1478.33, 540.78],
                     [0, 0, 1.0]])

# nominal target lengths (m) -> sonar box margins; category ids follow
# uxo_to_paired.CATEGORIES: 100lbs_aircraft_bomb, incindiary_grenade,
# mortar_shell
CLASS_LENGTH_M = {1: 1.20, 2: 0.40, 3: 0.55}

_SIG_ANG = 0.0087   # 0.5 deg: box-centre + blob-centroid noise (rad)
_SIG_RNG = 0.03     # blob range noise (m)
_SIG_B_PRIOR = 0.25  # b_rng prior sigma: bounded echo offset (front
                     # face vs box centre + sound-speed residual, m)
_B_RNG_MAX = 0.8    # hard bound on b_rng: |echo offset| <= half the
                    # longest target (0.6 m bomb) + sound-speed slack.
                    # Without it the (P_radial, b) degeneracy -- flat
                    # range profile when the target sits near the arc
                    # centre, the real tank layout -- lets the joint fit
                    # run away (observed: b = -16 m on 100927)
_GATE_COL = 8.0     # blob-vs-projection consistency gate (beams)
_GATE_RNG = 0.20    # blob-vs-projection consistency gate (m)
_OUTLIER_AZ_DEG = 6.0   # calibration blob rejection gate
_OUTLIER_RNG_M = 0.5


def _quat2r(q: Sequence[float]) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, float)).as_matrix()


def _chain(*tf: Tuple[Sequence[float], Sequence[float]]):
    """(R, p) of the last frame expressed in the first frame."""
    R, p = np.eye(3), np.zeros(3)
    for t, q in tf:
        p = p + R @ np.asarray(t, float)
        R = R @ _quat2r(q)
    return R, p


_R_PC_AR3, _P_AR3_PC = _chain(_T_PC_AR3)
_R_AR3_CAM, _P_CAM_AR3 = _chain(_T_HOLE_AR3, _T_AXIS_HOLE, _T_CAM_AXIS)
_, _P_SON_AR3 = _chain(_T_ARIS_AR3, _T_SONAR_ARIS)
_R_INIT = _R_PC_AR3 @ _R_AR3_CAM   # static-chain orientation (head aim off)
_R_SON_INIT = _R_INIT @ np.diag([1.0, -1.0, -1.0])  # empirical start


def _centers(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Camera / sonar origins in odom for a gantry translation g."""
    ar3 = np.asarray(g, float) + _P_AR3_PC
    return ar3 + _R_PC_AR3 @ _P_CAM_AR3, ar3 + _R_PC_AR3 @ _P_SON_AR3


def detect_blobs(img: np.ndarray) -> List[Tuple[float, float, float, float,
                                                float]]:
    """Bright compact returns: [(peak, row_c, col_c, w, h)], best first."""
    im = img.astype(np.float32)
    mask = (im >= np.percentile(im, 99.5)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20 or w < 4 or h < 4 or w > 45 or h > 70:
            continue
        sel = lab[y:y + h, x:x + w] == i
        peak = float(im[y:y + h, x:x + w][sel].max())
        out.append((peak, y + h / 2.0, x + w / 2.0, float(w), float(h)))
    out.sort(reverse=True)
    return out


def fit_circle(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Kasa least-squares circle fit: (centre(2,), radial rms residual)."""
    A = np.c_[x, y, np.ones_like(x)]
    b = x ** 2 + y ** 2
    a = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy = a[0] / 2, a[1] / 2
    r = np.hypot(x - cx, y - cy)
    return np.array([cx, cy]), float(np.sqrt(np.mean((r - r.mean()) ** 2)))


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------
@dataclass
class FrameObs:
    frame_idx: int
    sample_id: str
    gantry: np.ndarray
    window_start: float
    sample_period: float
    soundspeed: float
    cam_ray: Optional[np.ndarray] = None    # unit bearing, camera frame
    cam_ann_id: Optional[int] = None
    category_id: Optional[int] = None
    blob: Optional[Tuple[float, float, float, float]] = None  # r,c,w,h
    theta: float = 0.0                      # blob beam angle (rad)
    rng: float = 0.0                        # blob slant range (m)

    @property
    def dr(self) -> float:
        return self.sample_period * 1e-6 * self.soundspeed / 2


def load_observations(paired_root: str) -> Dict[str, List[FrameObs]]:
    """Per-recording frame observations from a paired-format root.

    Reads frame_meta.csv (gantry pose, window, sound speed, sample
    period -- the converter carries all of it), camera.json for GT
    bearings, and detects the target blob on every sonar frame (the
    strongest size-filtered bright component; calibration rejects
    mispicks as outliers).
    """
    with open(os.path.join(paired_root, "camera.json"),
              encoding="utf-8") as f:
        cam = json.load(f)
    ann_by_img: Dict[int, List[dict]] = {}
    for a in cam["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    recs: Dict[str, List[FrameObs]] = {}
    with open(os.path.join(paired_root, "frame_meta.csv"),
              newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            o = FrameObs(
                frame_idx=int(r["frame_idx"]),
                sample_id=r["sample_id"],
                gantry=np.array([float(r["gantry_x"]),
                                 float(r["gantry_y"]),
                                 float(r["gantry_z"])]),
                window_start=float(r["aris_windowstart"]),
                sample_period=float(r["aris_sampleperiod"]),
                soundspeed=float(r["aris_soundspeed"]))
            for a in ann_by_img.get(int(r["camera_image_id"]), []):
                bx, by, bw, bh = a["bbox"]
                u, v = bx + bw / 2, by + bh / 2
                ray = np.array([(u - K_LINEAR[0, 2]) / K_LINEAR[0, 0],
                                (v - K_LINEAR[1, 2]) / K_LINEAR[1, 1], 1.0])
                o.cam_ray = ray / np.linalg.norm(ray)
                o.cam_ann_id = a["id"]
                o.category_id = a["category_id"]
                break
            img = cv2.imread(os.path.join(
                paired_root, "images", "sonar",
                f"{r['sample_id']}.pgm"), cv2.IMREAD_UNCHANGED)
            if img is not None:
                blobs = detect_blobs(img)
                if blobs:
                    _, row_c, col_c, _, _ = blobs[0]
                    o.blob = (row_c, col_c, blobs[0][3], blobs[0][4])
                    o.theta = float(np.radians(beam_center_deg(col_c)))
                    o.rng = range_of_row(row_c, o.window_start,
                                         o.sample_period, o.soundspeed)
            recs.setdefault(r["recording"], []).append(o)
    return recs


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
@dataclass
class CalibResult:
    recording: str
    yaw_sign: int                      # +1/-1 arc-polar, 0 = constant
                                      # rotation, _YAW_AIM = aimed head
    circle_centre: np.ndarray
    phi_ref: float                     # arc angle of frame 0 (rad)
    yaw_span_deg: float
    R_cam0: np.ndarray                 # free camera rotation at phi_ref
    R_son0: np.ndarray
    P: np.ndarray                      # target position (odom, m)
    b_rng: float                       # sonar range bias (m)
    n_cam: int
    n_sonar: int
    rms_cam_deg: float
    rms_az_deg: float
    rms_rng_m: float


_YAW_AIM = 2    # head continuously re-aimed at the target


def _aim_rotmats(dirs: np.ndarray) -> np.ndarray:
    """Rodrigues rotations (n,3,3) mapping +z onto each unit row of
    ``dirs`` -- the bracket orientation of an aimed head (bracket->world).
    """
    a = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    b = np.array([0.0, 0.0, 1.0])
    v = np.cross(np.broadcast_to(b, a.shape), a)          # axes * sin
    s = np.linalg.norm(v, axis=1)                         # sin(angle)
    c = a @ b                                             # cos(angle)
    K = np.zeros((len(a), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -v[:, 2], v[:, 1]
    K[:, 1, 0], K[:, 1, 2] = v[:, 2], -v[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -v[:, 1], v[:, 0]
    # Rodrigues: R = I + s*K_hat + (1-c)*K_hat^2 with K_hat the skew of
    # the unit axis (K above is axis*sin, so divide by s; guard s~0,
    # where the frame is already aligned and R ~ I)
    small = s < 1e-8
    K[~small] /= s[~small, None, None]
    K2 = np.einsum("nij,njk->nik", K, K)
    R = (np.eye(3)[None] + K * np.where(small, 0.0, s)[:, None, None]
         + K2 * (1 - np.where(small, 1.0, c))[:, None, None])
    return R


def _aim_rots_T(P: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """World->bracket rotations (n,3,3) of an aimed head per origin."""
    return np.transpose(_aim_rotmats(P - origins), (0, 2, 1))


def _frame_yaw(yaw_sign: int, gantry: np.ndarray,
               circle_centre: np.ndarray, phi_ref: float,
               P: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-frame bracket->world rotation of the head.

    yaw_sign 0 = straight traverse (constant rotation), +1/-1 = the
    trolley yaws with the fitted arc's polar angle, ``_YAW_AIM`` = the
    head is re-aimed at the target at P every frame (needs P).
    """
    if yaw_sign == _YAW_AIM:
        los = P - _centers(gantry)[0]
        return _aim_rotmats(los[None])[0]
    if yaw_sign == 0:
        return np.eye(3)
    phi = np.arctan2(gantry[1] - circle_centre[1],
                     gantry[0] - circle_centre[0])
    return Rotation.from_euler(
        "z", yaw_sign * (phi - phi_ref)).as_matrix()


def _staged_fit(cam: List[FrameObs], son: List[FrameObs],
                C: np.ndarray, phi_ref: float, yaw_sign: int,
                p_hyps: Sequence[np.ndarray]):
    """Robust three-stage fit; joint least squares from random-ish
    starts lands in local minima (verified on the synthetic tank).

    Stage 1 fits [R_cam, P] on camera bearings alone over a coarse
    rotation grid of static-chain and look-at seeds (the head aim is
    unknown; the static chain alone is not a good enough seed).
    ``p_hyps`` are candidate target positions (from the mean observed
    sonar range) that seed the look-at boresights.  For the ``aim``
    convention the per-frame rotations depend on P and are rebuilt
    inside each residual; its stage-1 P seeds are the ``p_hyps``
    themselves (an aimed head observes rays that converge on P, so
    those seeds converge without boresight search).  Stage 2 fits
    [R_son, b_rng] on sonar azimuths+ranges with (R_cam, P) held.
    Stage 3 refines all ten jointly.  Returns the scipy result of
    stage 3.
    """
    fixed_cam = None if yaw_sign == _YAW_AIM else np.array(
        [_frame_yaw(yaw_sign, o.gantry, C, phi_ref).T for o in cam])
    fixed_son = None if yaw_sign == _YAW_AIM else np.array(
        [_frame_yaw(yaw_sign, o.gantry, C, phi_ref).T for o in son])
    c_cam = np.array([_centers(o.gantry)[0] for o in cam])
    c_son = np.array([_centers(o.gantry)[1] for o in son])
    # the head is one rigid body aimed by the CAMERA: the align
    # reference is the camera origin on the sonar frames too (the
    # ~0.3 m camera-sonar baseline would otherwise enter as a
    # parallax-level azimuth bias)
    c_aim_son = np.array([_centers(o.gantry)[0] for o in son])
    A = np.array([o.cam_ray for o in cam])
    TH = np.array([o.theta for o in son])
    RG = np.array([o.rng for o in son])
    rng0 = float(np.mean(RG))

    def rz_cam(P):
        return fixed_cam if fixed_cam is not None else _aim_rots_T(P, c_cam)

    def rz_son(P):
        if fixed_son is not None:
            return fixed_son
        return _aim_rots_T(P, c_aim_son)

    def cam_resid(p):
        Rcam = Rotation.from_rotvec(p[:3]).as_matrix()
        d = np.einsum("nij,nj->ni", rz_cam(p[3:6]), p[3:6] - c_cam)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        cosang = np.clip(np.einsum("ni,ni->n", A, d @ Rcam), -1, 1)
        return np.arccos(cosang) / _SIG_ANG

    seeds = [_R_INIT]
    for yaw in (90, 180, 270):
        for pitch in (-60, -30, 0, 30):
            seeds.append(_R_INIT @ Rotation.from_euler(
                "zyx", [yaw, pitch, 0], degrees=True).as_matrix())
    # look-at seeds: the operator keeps the target in view, so also
    # aim boresights at the hypothesized target positions (the
    # static-chain seed can be >100 deg off, e.g. the synthetic tank,
    # and stage 1 then lands in a wrong-basin local minimum -- which
    # the camera-only cost cannot distinguish, so always run them all)
    c_bar = c_cam.mean(axis=0)
    for p_hyp in p_hyps:
        for c_anchor in (c_cam[0], c_bar, c_cam[-1]):
            d_aim = p_hyp - c_anchor
            d_aim /= np.linalg.norm(d_aim)
            r_look = Rotation.align_vectors(
                [d_aim], [[0.0, 0.0, 1.0]])[0].as_matrix()
            for roll in (0, 90, 180, 270):
                seeds.append(r_look @ Rotation.from_euler(
                    "z", roll, degrees=True).as_matrix())
    starts = [(R0, np.mean([c_cam[i] + rng0 * (R0 @ A[i])
                            for i in range(len(cam))], axis=0))
              for R0 in seeds]
    if yaw_sign == _YAW_AIM:
        # aimed head: rays converge on P; mis-aim is a small constant
        # rotation about the (near-identity) mapping, so seed R with
        # identity rolls and P with the position hypotheses
        starts = [(np.eye(3) @ Rotation.from_euler(
            "z", roll, degrees=True).as_matrix(), p_hyp)
            for p_hyp in p_hyps for roll in (0, 180)]
        starts.append((_R_INIT, p_hyps[0]))
    best = None
    for R0, P0 in starts:
        r = least_squares(cam_resid,
                          np.r_[Rotation.from_matrix(R0).as_rotvec(), P0],
                          loss="huber", f_scale=2.0)
        if best is None or r.cost < best.cost:
            best = r
    Rcam0 = Rotation.from_rotvec(best.x[:3]).as_matrix()
    P0 = best.x[3:6]
    Rz_son = rz_son(P0)

    def son_resid(q):
        Rson = Rotation.from_rotvec(q[:3]).as_matrix()
        v = P0 - c_son
        vs = np.einsum("nij,nj->ni", Rz_son, v) @ Rson
        dth = np.arctan2(vs[:, 1], vs[:, 0]) - TH
        r_res = np.linalg.norm(v, axis=1) + q[3] - RG
        return np.r_[np.arctan2(np.sin(dth), np.cos(dth)) / _SIG_ANG,
                     r_res / _SIG_RNG]

    q0 = np.r_[Rotation.from_matrix(
        Rcam0 @ np.diag([1.0, -1.0, -1.0])).as_rotvec(), 0.0]
    r2 = least_squares(son_resid, q0, loss="huber", f_scale=2.0,
                       x_scale=np.array([1.0, 1.0, 1.0, 0.05]),
                       bounds=(np.r_[np.full(3, -np.inf), -_B_RNG_MAX],
                               np.r_[np.full(3, np.inf), _B_RNG_MAX]),
                       max_nfev=5000)

    # stage 3: full joint residual, seeded from stages 1-2
    def joint_resid(p):
        Rcam = Rotation.from_rotvec(p[0:3]).as_matrix()
        Rson = Rotation.from_rotvec(p[7:10]).as_matrix()
        P, br = p[3:6], p[6]
        d = np.einsum("nij,nj->ni", rz_cam(P), P - c_cam)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        cosang = np.clip(np.einsum("ni,ni->n", A, d @ Rcam), -1, 1)
        v = P - c_son
        vs = np.einsum("nij,nj->ni", rz_son(P), v) @ Rson
        dth = np.arctan2(vs[:, 1], vs[:, 0]) - TH
        r_res = np.linalg.norm(v, axis=1) + br - RG
        # weak prior on b_rng: it models a bounded physical effect
        # (front-face echo vs box centre, ~0.1 m).  Without it a
        # straight approach pass is exactly degenerate -- shifting P
        # along the path moves every range uniformly, which b absorbs
        return np.r_[np.arccos(cosang) / _SIG_ANG,
                     np.arctan2(np.sin(dth), np.cos(dth)) / _SIG_ANG,
                     r_res / _SIG_RNG,
                     br / _SIG_B_PRIOR]

    p0 = np.r_[best.x[:6], r2.x[3], r2.x[:3]]
    # x_scale/max_nfev: with the default budget (100*nfev) the numeric
    # jacobian runs out of evaluations before converging on the
    # mildly ill-conditioned b_rng <-> P-radial direction
    bnds = (np.r_[np.full(6, -np.inf), -_B_RNG_MAX, np.full(3, -np.inf)],
            np.r_[np.full(6, np.inf), _B_RNG_MAX, np.full(3, np.inf)])
    return least_squares(joint_resid, p0, loss="huber", f_scale=2.0,
                         x_scale=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                           0.05, 1.0, 1.0, 1.0]),
                         bounds=bnds, max_nfev=10000)


def calibrate(recording: str, obs: List[FrameObs],
              min_frames: int = 20) -> CalibResult:
    """Fit the per-recording calibration; drops outlier blobs once."""
    cam = [o for o in obs if o.cam_ray is not None]
    son = [o for o in obs if o.blob is not None]
    if len(cam) < min_frames or len(son) < min_frames // 2:
        raise ValueError(f"{recording}: not enough observations "
                         f"({len(cam)} camera / {len(son)} sonar)")

    G = np.array([o.gantry for o in obs])
    C, circ_resid = fit_circle(G[:, 0], G[:, 1])
    phi = np.arctan2(G[:, 1] - C[1], G[:, 0] - C[0])
    phi_ref = float(phi[0])
    # arc vs straight-line traverse: a Kasa fit through a noisy line
    # returns a bogus finite circle, so decide by how much better the
    # circle explains the path than a straight line (PCA minor axis)
    xy = G[:, :2] - G[:, :2].mean(axis=0)
    line_resid = float(np.linalg.svd(xy, compute_uv=False)[1]
                       / np.sqrt(len(xy)))
    arc_like = circ_resid < 0.3 * line_resid
    if not arc_like:
        phi = np.zeros_like(phi)
        span = 0.0
    else:
        span = float(np.degrees(phi.max() - phi.min()))
    signs = [0, _YAW_AIM] if not arc_like else [1, -1, _YAW_AIM]

    # candidate target positions for the look-at seeds, derived from
    # the observed mean sonar range rng0: near the arc centre (arc
    # traverses) or abeam of the path (straight traverses)
    rng0 = float(np.mean([o.rng for o in son]))
    z_bar = float(G[:, 2].mean())
    p_hyps = []
    if arc_like:
        r_arc = float(np.mean(np.linalg.norm(G[:, :2] - C, axis=1)))
        drop = np.sqrt(max(rng0 ** 2 - r_arc ** 2, 0.04))
        for f in (0.7, 1.0, 1.3):
            p_hyps.append(np.array([C[0], C[1], z_bar - f * drop]))
    else:
        n_hat = np.array([-(G[-1, 1] - G[0, 1]), G[-1, 0] - G[0, 0]])
        n_hat /= np.linalg.norm(n_hat)
        mid = G[:, :2].mean(axis=0)
        for side in (1.0, -1.0):
            for h in (0.15, 0.4, 0.8):   # real flybys pass close by
                d = side * h * rng0
                drop = np.sqrt(max(rng0 ** 2 - d ** 2, 0.04))
                p_hyps.append(np.array([mid[0] + n_hat[0] * d,
                                        mid[1] + n_hat[1] * d,
                                        z_bar - drop]))

    r = None
    for s in signs:                      # pick the better yaw convention
        ri = _staged_fit(cam, son, C, phi_ref, s, p_hyps)
        if r is None or ri.cost < r.cost:
            r, yaw_sign = ri, s

    # one outlier-rejection round on sonar blobs (mispicked echoes)
    def _keep_mask(Rson):
        keep = []
        for o in son:
            Rz = _frame_yaw(yaw_sign, o.gantry, C, phi_ref, P=r.x[3:6])
            v = r.x[3:6] - _centers(o.gantry)[1]
            vs = Rson.T @ (Rz.T @ v)
            dth = np.degrees(np.arctan2(vs[1], vs[0]) - o.theta)
            keep.append(abs(dth) < _OUTLIER_AZ_DEG
                        and abs(np.linalg.norm(v) + r.x[6] - o.rng)
                        < _OUTLIER_RNG_M)
        return np.array(keep)

    Rson = Rotation.from_rotvec(r.x[7:10]).as_matrix()
    keep = _keep_mask(Rson)
    if (keep.sum() >= max(min_frames // 2, 10)) and keep.sum() < len(keep):
        son2 = [o for o, k in zip(son, keep) if k]
        for s in signs:
            ri = _staged_fit(cam, son2, C, phi_ref, s, p_hyps)
            # compare mean (per-residual) cost: the refit sees fewer
            # frames, so total cost would drop mechanically
            if ri.cost / ri.fun.size < r.cost / r.fun.size:
                r, yaw_sign, son = ri, s, son2

    Rcam = Rotation.from_rotvec(r.x[0:3]).as_matrix()
    Rson = Rotation.from_rotvec(r.x[7:10]).as_matrix()
    cerr, aerr, rerr = [], [], []
    for o in cam:
        Rz = _frame_yaw(yaw_sign, o.gantry, C, phi_ref, P=r.x[3:6])
        d = Rz.T @ (r.x[3:6] - _centers(o.gantry)[0])
        cerr.append(np.degrees(np.arccos(np.clip(
            o.cam_ray @ (Rcam.T @ d) / np.linalg.norm(d), -1, 1))))
    for o in son:
        Rz = _frame_yaw(yaw_sign, o.gantry, C, phi_ref, P=r.x[3:6])
        vs = Rson.T @ (Rz.T @ (r.x[3:6] - _centers(o.gantry)[1]))
        aerr.append(np.degrees(np.arctan2(vs[1], vs[0]) - o.theta))
        rerr.append(np.linalg.norm(r.x[3:6] - _centers(o.gantry)[1])
                    + r.x[6] - o.rng)

    def _rms(x):
        x = np.asarray(x)
        return float(np.sqrt(np.mean(x ** 2))) if x.size else 0.0

    return CalibResult(recording=recording, yaw_sign=yaw_sign,
                       circle_centre=C, phi_ref=phi_ref,
                       yaw_span_deg=span, R_cam0=Rcam, R_son0=Rson,
                       P=r.x[3:6], b_rng=float(r.x[6]), n_cam=len(cam),
                       n_sonar=len(son), rms_cam_deg=_rms(cerr),
                       rms_az_deg=_rms(aerr), rms_rng_m=_rms(rerr))


# --------------------------------------------------------------------------
# projection / GT generation
# --------------------------------------------------------------------------
def predict(calib: CalibResult, o: FrameObs) -> Tuple[float, float]:
    """Projected (row_c, col_c) of the calibrated target in frame o."""
    Rz = _frame_yaw(calib.yaw_sign, o.gantry, calib.circle_centre,
                    calib.phi_ref, P=calib.P)
    v = calib.P - _centers(o.gantry)[1]
    vs = calib.R_son0.T @ (Rz.T @ v)
    row = row_of_range(np.linalg.norm(v) + calib.b_rng, o.window_start,
                       o.sample_period, o.soundspeed)
    col = col_of_angle(np.degrees(np.arctan2(vs[1], vs[0])))
    return float(row), float(col)


def sonar_bbox(calib: CalibResult, o: FrameObs) -> Optional[Dict]:
    """GT entry for one frame: bbox + source + pi link + diagnostics.

    Boxes are clipped to the 128x441 image; a frame whose (clipped)
    box is empty -- the calibrated projection lands outside the
    insonified window and no gated blob vouches for the target --
    yields no GT rather than a fabricated box (returns None).
    """
    row_p, col_p = predict(calib, o)
    L = CLASS_LENGTH_M.get(o.category_id, 0.5)
    rng = max(float(np.linalg.norm(calib.P - _centers(o.gantry)[1])), 0.3)
    dth = np.degrees(np.arctan2(L / 2, rng))
    col_lo = col_of_angle(beam_center_deg(col_p) - dth)
    col_hi = col_of_angle(beam_center_deg(col_p) + dth)
    row_lo = row_p - (L / 2 + 0.04) / o.dr
    row_hi = row_p + (L / 2 + 0.04) / o.dr
    source = "projected"
    if o.blob is not None:
        br, bc, bw, bh = o.blob
        b_rng = range_of_row(br, o.window_start, o.sample_period,
                             o.soundspeed)
        if (abs(bc - col_p) <= _GATE_COL
                and abs(b_rng - (rng + calib.b_rng)) <= _GATE_RNG):
            source = "blob"
            col_lo, col_hi = bc - bw / 2 - 1, bc + bw / 2 + 1
            row_lo, row_hi = br - bh / 2 - 1, br + bh / 2 + 1
    x0, y0 = max(col_lo, 0.0), max(row_lo, 0.0)
    x1, y1 = min(col_hi, 128.0), min(row_hi, 441.0)
    if x1 - x0 < 2.0 or y1 - y0 < 2.0:
        return None
    bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
    return dict(bbox=bbox, category_id=o.category_id, source=source,
                cam_ann_id=o.cam_ann_id, projected=(row_p, col_p),
                blob=o.blob,
                blob_delta=(abs(o.blob[1] - col_p),
                            abs(o.blob[0] - row_p)) if o.blob else None)


def project_uxo_sonar(paired_root: str, calib: CalibResult,
                      obs: List[FrameObs]) -> Dict[str, Dict]:
    """Sonar GT entries for one recording: {sample_id: sonar_bbox()}.

    Frames whose sonar_bbox is None (projection outside the window,
    no gated blob) are left out -- they get no sonar annotation.
    """
    out = {}
    for o in obs:
        if o.category_id is None:
            continue
        entry = sonar_bbox(calib, o)
        if entry is not None:
            out[o.sample_id] = entry
    return out
