"""Tests for the UXO sonar projection calibration (synthetic tank).

Builds a synthetic gantry arc + target + camera/sonar observations from
known geometry, runs the full calibrate->predict->bbox pipeline, and
asserts recovery.  No real-data dependency: the real sample acceptance
is the contact-sheet eyeball + projection report (F2 TODO 14.2).
"""

import json
import os
import sys

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aodet.datasets.converters import uxo_projection as up
from aodet.geometry.aris_beam_table import (beam_center_deg,
                                            col_of_angle,
                                            range_of_row,
                                            row_of_range)

RNG = np.random.default_rng(7)


# --------------------------------------------------------------------------
# beam table / geometry helpers
# --------------------------------------------------------------------------
def test_beam_table_endpoints_and_roundtrip():
    c = np.array([t[0] for t in
                  __import__("aodet.geometry.aris_beam_table",
                             fromlist=["BEAM_WIDTHS_ARIS3000_128"])
                  .BEAM_WIDTHS_ARIS3000_128])
    assert len(c) == 128
    assert c[0] == pytest.approx(-15.0068, abs=1e-4)
    assert c[-1] == pytest.approx(15.0156, abs=1e-4)
    assert (np.diff(c) > 0).all()
    # fractional column round trip across the fan
    for col in (0.0, 5.5, 64.0, 100.25, 127.0):
        assert col_of_angle(beam_center_deg(col)) == pytest.approx(
            col, abs=1e-3)


def test_range_row_mapping():
    ws, per, c = 0.6949, 4.0, 1491.28
    dr = per * 1e-6 * c / 2
    assert dr == pytest.approx(0.002983, abs=1e-5)
    assert range_of_row(0, ws, per, c) == pytest.approx(ws)
    assert row_of_range(range_of_row(440, ws, per, c), ws, per, c) == \
        pytest.approx(440)


# --------------------------------------------------------------------------
# blob detector + circle fit
# --------------------------------------------------------------------------
def test_detect_blobs_finds_target_not_band():
    img = np.full((441, 128), 30.0)
    img[200:260, :] = 110.0                       # floor band (wide)
    img += RNG.normal(0, 12, img.shape)           # sonar texture
    rr, cc = np.mgrid[0:441, 0:128]
    img += 200 * np.exp(-((rr - 220) ** 2 + (cc - 57) ** 2) / 12.0)
    blobs = up.detect_blobs(np.clip(img, 0, 255).astype(np.uint8))
    assert blobs, "no blob detected"
    peak, row_c, col_c, w, h = blobs[0]
    assert abs(row_c - 220) <= 2 and abs(col_c - 57) <= 2
    assert 3 <= w <= 20 and 3 <= h <= 20


def test_fit_circle_recovers_centre():
    t = np.radians(np.linspace(-160, -20, 60))
    cx, cy, r = 1.2, 1.4, 1.08
    x = cx + r * np.cos(t) + RNG.normal(0, 2e-3, t.size)
    y = cy + r * np.sin(t) + RNG.normal(0, 2e-3, t.size)
    C, resid = up.fit_circle(x, y)
    assert np.allclose(C, [cx, cy], atol=0.01)
    assert resid < 0.01


# --------------------------------------------------------------------------
# synthetic tank: full pipeline
# --------------------------------------------------------------------------
def _synth_tank(arc=True, n=100, b_true=0.06, aim=False):
    """Frames + observations from known geometry (mirrors the model).

    ``aim=False``: polar/constant head (tangent-locked trolley).
    ``aim=True``: the head is re-aimed at the target every frame with
    a small operator wobble -- the convention the real arc recordings
    follow (target stays near the image centre for the whole traverse).

    Uses a per-case seeded RNG so thresholds are stable regardless of
    test execution order (a shared module-level stream made results
    realization-dependent).
    """
    gen = np.random.default_rng(11 if aim else 7)
    C_true = np.array([1.2, 1.4])
    if arc:
        # like the real tank: the target sits near the arc centre, so
        # it stays inside the sonar fan for the full ~130 deg traverse
        phi = np.radians(np.linspace(-160, -30, n))
        G = np.c_[C_true[0] + 1.08 * np.cos(phi),
                  C_true[1] + 1.08 * np.sin(phi),
                  np.full(n, -0.25)]
        P_true = np.array([1.32, 1.33, -1.15])
    else:
        # straight flyby (mirrors the real 105856/124735): a 0.6 m
        # traverse at ~0.17 m lateral miss from the target xy, gantry
        # heading fixed (yaw_sign=0)
        P_true = np.array([1.20, 1.42, -1.20])
        G = np.c_[np.linspace(0.35, 0.95, n), np.full(n, 1.25),
                  np.full(n, -0.25)]
        G += np.c_[gen.normal(0, 1e-3, n), gen.normal(0, 1e-3, n),
                   gen.normal(0, 1e-3, n)]
    # aim camera z at the target from the first frame, small mount error;
    # sonar boresight (x) likewise, with its own fixed relative rotation
    c0, s0 = up._centers(G[0] if arc else G[n // 2])
    d0 = (P_true - c0) / np.linalg.norm(P_true - c0)
    R_look = Rotation.align_vectors([d0], [[0.0, 0.0, 1.0]])[0].as_matrix()
    R_free_cam = R_look @ Rotation.from_euler(
        "xyz", [3, -2, 8], degrees=True).as_matrix()
    R_look_s = Rotation.align_vectors([d0], [[1.0, 0.0, 0.0]])[0].as_matrix()
    R_free_son = R_look_s @ Rotation.from_euler(
        "xyz", [1, -5, 6], degrees=True).as_matrix()
    if aim:
        # free rotations live in the ALIGNED frame (z = line of sight):
        # camera z and sonar x sit on the boresight with mount errors
        R_free_cam = Rotation.from_euler(
            "xyz", [3, -2, 8], degrees=True).as_matrix()
        R_free_son = Rotation.align_vectors(
            [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])[0].as_matrix() \
            @ Rotation.from_euler("xyz", [1, -5, 6], degrees=True).as_matrix()
    yaw_sign_true = (up._YAW_AIM if aim else (1 if arc else 0))

    ws, per, c_sound = 0.6949, 4.0, 1491.28
    obs = []
    for i in range(n):
        Rz = np.eye(3)
        if aim:
            d_i = P_true - up._centers(G[i])[0]
            Rz = Rotation.align_vectors(
                [d_i / np.linalg.norm(d_i)], [[0.0, 0.0, 1.0]])[0].as_matrix()
            Rz = Rz @ Rotation.from_euler("yx", gen.normal(
                0, 0.4, 2), degrees=True).as_matrix()
        elif yaw_sign_true:
            Rz = Rotation.from_euler("z", phi[i] - phi[0]).as_matrix()
        R_cam_odom = Rz @ R_free_cam
        R_son_odom = Rz @ R_free_son
        c_i, s_i = up._centers(G[i])
        # camera pixel of the target (skip if behind)
        ray = R_cam_odom.T @ (P_true - c_i)
        ray /= np.linalg.norm(ray)
        if ray[2] <= 0.1:
            continue
        u = up.K_LINEAR[0, 0] * ray[0] / ray[2] + up.K_LINEAR[0, 2]
        v = up.K_LINEAR[1, 1] * ray[1] / ray[2] + up.K_LINEAR[1, 2]
        u += gen.normal(0, 2)
        v += gen.normal(0, 2)
        ray2 = np.array([(u - up.K_LINEAR[0, 2]) / up.K_LINEAR[0, 0],
                         (v - up.K_LINEAR[1, 2]) / up.K_LINEAR[1, 1], 1.0])
        # sonar frame geometry of the target
        vs = R_son_odom.T @ (P_true - s_i)
        r_slant = float(np.linalg.norm(P_true - s_i))
        theta = float(np.arctan2(vs[1], vs[0]))
        row = row_of_range(r_slant + b_true, ws, per, c_sound)
        col = float(col_of_angle(np.degrees(theta)))
        if not (0 <= row < 440 and 2 <= col <= 125):
            continue
        # synthesize the sonar image + detect the blob through the module
        img = np.full((441, 128), 30.0, np.float32)
        img[200:260, :] = 110.0
        img += gen.normal(0, 12, img.shape).astype(np.float32)
        rr, cc = np.mgrid[0:441, 0:128]
        img = img + 205 * np.exp(-((rr - row) ** 2 + (cc - col) ** 2) / 12.0)
        blobs = up.detect_blobs(np.clip(img, 0, 255).astype(np.uint8))
        o = up.FrameObs(
            frame_idx=i, sample_id=f"rec_{i:04d}", gantry=G[i],
            window_start=ws, sample_period=per, soundspeed=c_sound,
            cam_ray=ray2 / np.linalg.norm(ray2), cam_ann_id=i + 1,
            category_id=3)
        if blobs:
            _, row_c, col_c, w, h = blobs[0]
            o.blob = (row_c, col_c, w, h)
            o.theta = float(np.radians(beam_center_deg(col_c)))
            o.rng = range_of_row(row_c, ws, per, c_sound)
        obs.append(o)
    return obs, dict(P=P_true, yaw=yaw_sign_true,
                     R_cam=R_free_cam, R_son=R_free_son,
                     C=C_true, b=b_true)


@pytest.mark.parametrize("arc,aim", [(True, False), (False, False),
                                     (True, True), (False, True)],
                         ids=["polar-arc", "polar-line",
                              "aim-arc", "aim-line"])
def test_calibrate_recovers_synthetic_tank(arc, aim):
    """Arc traverse: full geometric recovery.  Straight flyby (like
    the real 105856/124735): enough bearing curvature for a precise
    projection, but the along-path target position trades against
    b_rng (a shift along the path moves every range nearly uniformly;
    the b prior caps the slide), so P is asserted loosely there.
    The arc has the same trade radially: with the target near the arc
    centre -- the real tank layout, kept here -- the range profile is
    nearly flat, so P_radial and b_rng are separated by the prior
    alone and both carry a prior-width-sized error.
    ``aim`` cases generate an operator-tracked head -- the convention
    the real arc recordings follow; the fit must select it over the
    polar/constant candidates."""
    obs, truth = _synth_tank(arc=arc, aim=aim)
    assert len(obs) >= 40
    calib = up.calibrate("synthetic", obs, min_frames=20)
    assert calib.yaw_sign == truth["yaw"]
    assert calib.rms_cam_deg < 1.0
    assert calib.rms_az_deg < 1.0
    err_P = np.linalg.norm(calib.P - truth["P"])
    if arc:
        assert err_P < 0.15, f"P error {err_P:.3f} m"
        assert abs(calib.b_rng - truth["b"]) < 0.08
    else:
        assert err_P < 0.8, f"P error {err_P:.3f} m"
        assert abs(calib.b_rng - truth["b"]) < 0.15
    # the joint range prediction |P - s| + b is identified in every
    # geometry -- it is the quantity the sonar box rows depend on
    assert calib.rms_rng_m < 0.05

    # projection lands on the synthesised blob for every observed frame.
    # aim cases carry the per-frame operator wobble (0.4 deg yaw/pitch,
    # unmodelled by the constant-misaim fit), so their max azimuth
    # residual is the wobble extreme -- ~1.2 deg at the fan edge where
    # beams are ~0.23 deg apart -- not a fit error; still well inside
    # the 8-beam blob gate used for GT source selection
    dcol, drow = [], []
    for o in obs:
        if o.blob is None:
            continue
        row_p, col_p = up.predict(calib, o)
        dcol.append(abs(col_p - o.blob[1]))
        drow.append(abs(row_p - o.blob[0]))
    assert dcol and max(dcol) < (2.5 if arc and not aim else
                                 3.0 if not aim else 6.0)        # beams
    assert max(drow) < 4.0                                        # rows

    # GT box policy: observed blob within the gate -> source="blob";
    # fully-out-of-window projections are dropped entirely (None)
    boxes = [b for b in (up.sonar_bbox(calib, o) for o in obs)
             if b is not None]
    assert len(boxes) >= 0.9 * len(obs)
    sources = [b["source"] for b in boxes]
    assert sources.count("blob") >= 0.9 * len(boxes)
    for b in boxes:
        x, y, w, h = b["bbox"]
        assert 0 <= x and x + w <= 128 and 0 <= y and y + h <= 441
        assert w >= 3 and h >= 3


def test_calibrate_needs_min_frames():
    obs, _ = _synth_tank(n=30)
    with pytest.raises(ValueError):
        up.calibrate("short", obs[:5], min_frames=20)


def test_anchor_b_rng_recovers_range_offset():
    """E05 regression: a b_rng the degeneracy let slide into P shows up
    as a systematic (blob_row - projected_row) offset; the anchor must
    put it back in one median step, exactly (predict's row is affine in
    b_rng)."""
    obs, truth = _synth_tank(arc=False, aim=False)
    calib = up.calibrate("synthetic", obs, min_frames=20)
    from dataclasses import replace as _replace
    biased = _replace(calib, b_rng=calib.b_rng - 0.30)  # the slide
    drow = [abs(up.predict(biased, o)[0] - o.blob[0])
            for o in obs if o.blob is not None]
    assert np.median(drow) > 0.25 / obs[0].dr * 0.5     # visibly off
    fixed, stats = up.anchor_b_rng(biased, obs)
    assert stats["n_anchor"] >= 10
    assert abs(fixed.b_rng - calib.b_rng) < 0.03        # recovered
    drow = [abs(up.predict(fixed, o)[0] - o.blob[0])
            for o in obs if o.blob is not None]
    assert max(drow) < 4.0                              # rows, as calibrated


def test_anchor_b_rng_keeps_b_when_no_blobs():
    """Too few bearing-gated blobs: b_rng must pass through untouched."""
    obs, _ = _synth_tank(arc=False, aim=False, n=40)
    calib = up.calibrate("synthetic", obs, min_frames=20)
    stripped = [type(o)(frame_idx=o.frame_idx, sample_id=o.sample_id,
                        gantry=o.gantry, window_start=o.window_start,
                        sample_period=o.sample_period,
                        soundspeed=o.soundspeed) for o in obs[:5]]
    for o in stripped:
        o.cam_ray, o.category_id, o.blob = None, 3, None
    fixed, stats = up.anchor_b_rng(calib, stripped)
    assert fixed is calib and stats["n_anchor"] == 0
    assert stats["b_rng_after"] == stats["b_rng_before"]


def test_calib_cache_roundtrip():
    """tools/project_uxo_sonar.py cache helpers: a serialised ->
    deserialised CalibResult must reproduce predict() bit-for-bit."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "project_uxo_sonar",
        os.path.join(os.path.dirname(up.__file__), "..", "..", "..",
                     "tools", "project_uxo_sonar.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    obs, _ = _synth_tank(arc=False, aim=False)
    calib = up.calibrate("synthetic", obs, min_frames=20)
    back = tool._calib_from_dict(tool._calib_to_dict(calib))
    for o in obs[:10]:
        assert np.allclose(up.predict(calib, o), up.predict(back, o))
