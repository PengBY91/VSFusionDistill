"""Tests for tools/uxo_inventory.py (E11 prereq 4): category, path
geometry, blob-frame count, moving-target flag, and the projection-
report merge that supplies the calibration residuals the E11 标定质量
过滤 filters on.
"""

import csv
import importlib.util
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tool():
    spec = importlib.util.spec_from_file_location(
        "uxo_inventory",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "tools", "uxo_inventory.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


REC_ARC = "2023-01-05_100000"
REC_LINE = "2023-01-06_110000"
N = 30


def _make_rec(src, rec, *, arc, centers_x, blob_rows):
    """Recording with N labeled frames; centres and blob rows given as
    per-frame lists (len N, None = no label that frame)."""
    rng = np.random.default_rng(3)
    rd = src / rec
    for sub in ("labels", "aris_raw", "gopro"):
        (rd / sub).mkdir(parents=True)
    (rd / "aris_file_meta.yaml").write_text(
        "NumRawBeams: 128\nSamplesPerChannel: 441\n")
    t = np.linspace(0, 1, N)
    if arc:
        phi = np.radians(-160 + 130 * t)
        gx, gy = 1.2 + 1.08 * np.cos(phi), 1.4 + 1.08 * np.sin(phi)
    else:
        gx, gy = 0.35 + 0.6 * t, np.full(N, 1.25)
    gx += rng.normal(0, 1e-3, N)
    gy += rng.normal(0, 1e-3, N)
    for i in range(N):
        cx = centers_x[i]
        if cx is not None:
            (rd / "labels" / f"{i:04d}.json").write_text(json.dumps(
                {"class": "mortar_shell", "x_min": cx - 50,
                 "y_min": 100, "x_max": cx + 50, "y_max": 140}))
        cv2.imwrite(str(rd / "gopro" / f"{i:04d}.jpg"),
                    np.zeros((16, 16, 3), np.uint8))
        img = np.full((441, 128), 30.0, np.float32)
        img[200:260, :] = 110.0
        img += rng.normal(0, 12, img.shape).astype(np.float32)
        r, c = np.mgrid[0:441, 0:128]
        row = blob_rows[i]
        if row is not None:
            img = img + 205 * np.exp(-((r - row) ** 2 + (c - 64) ** 2) / 12.0)
        cv2.imwrite(str(rd / "aris_raw" / f"{i:04d}.pgm"),
                    np.clip(img, 0, 255).astype(np.uint8))
    with open(rd / "gantry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aris_frame_idx", "x", "y", "z"])
        for i in range(N):
            w.writerow([i, gx[i], gy[i], -0.25])
    with open(rd / "aris_frame_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FrameIndex", "WindowStart", "WindowLength",
                    "SamplePeriod", "SoundSpeed"])
        for i in range(N):
            w.writerow([i, 0.69, 2.32, 4.0, 1491.3])


def _make_src(tmp_path):
    src = tmp_path / "src"
    # arc + static camera centre + drifting sonar echo: sonar motion
    _make_rec(src, REC_ARC, arc=True,
              centers_x=[500 + 3 * np.sin(i / 5) for i in range(N)],
              blob_rows=[100 + 7 * i for i in range(N)])   # 100 -> 303
    # line + drifting camera centre + static echo: camera motion
    _make_rec(src, REC_LINE, arc=False,
              centers_x=[100 + 27 * i for i in range(N)],  # 100 -> 883
              blob_rows=[220 + 1.0 * np.sin(i / 3) for i in range(N)])
    return src


def test_inventory_paths_categories_blobs_moving(tmp_path):
    tool = _tool()
    src = _make_src(tmp_path)
    a = tool.inventory_recording(str(src / REC_ARC), REC_ARC)
    b = tool.inventory_recording(str(src / REC_LINE), REC_LINE)
    for e in (a, b):
        assert e["category"] == "mortar_shell"
        assert e["n_annotated"] == N
        assert e["n_blob_frames"] == N
        assert e["blob_ratio"] == 1.0
        assert not e["mixed_classes"]
        # difficult-subset draft axes: blob peak/bg contrast (fixture:
        # bg 30, blob peak ~235) and camera GT aspect (100x40 box)
        assert 6.0 < e["blob_contrast_median"] < 9.0
        assert e["gt_aspect_median"] == 2.5
    assert a["path"] == "arc"
    assert b["path"] == "line"
    # arc: camera centre static (sinusoid, no drift), sonar echo row
    # sweeps 100 -> 303 rows (45% of the 441-row range axis)
    assert a["moving_target_camera"] is False
    assert a["moving_target_sonar"] is True
    assert a["moving_target"] is True
    # line: camera centre sweeps 100 -> 883 in 640x360 space (x3 ->
    # 300 -> 2649 of 1920 px), echo static
    assert b["moving_target_camera"] is True
    assert b["moving_target_sonar"] is False
    assert b["moving_target"] is True


def test_short_or_steady_series_is_not_moving():
    tool = _tool()
    flat = [0.5] * 15
    assert tool._monotone_drift(flat) is False
    noisy = list(np.linspace(0.5, 0.52, 15) + np.linspace(0, 1e-3, 15))
    assert tool._monotone_drift(noisy) is False      # drift < 10%
    assert tool._monotone_drift([0.2, 0.21, 0.2, 0.5, 0.55, 0.6]) is False
    assert len(noisy) >= 10                          # sanity: long enough


def test_merge_projection_report_adds_head_and_residuals(tmp_path):
    tool = _tool()
    src = _make_src(tmp_path)
    inv = {"recordings": {
        REC_ARC: tool.inventory_recording(str(src / REC_ARC), REC_ARC)}}
    report = {REC_ARC: {
        "yaw_sign": 2, "rms_cam_deg": 0.5, "rms_az_deg": 0.8,
        "rms_rng_m": 0.03, "proj_err_row_px": [1.5, 4.0],
        "proj_err_col_beams": [2.0, 5.0]}}
    n = tool.merge_projection_report(inv, report)
    assert n == 1
    e = inv["recordings"][REC_ARC]
    assert e["head"] == "aim"
    assert e["geometry"] == "aim"        # aim is its own stratum
    assert e["path"] == "arc"            # path shape kept separately
    assert e["rms_az_deg"] == 0.8
    assert e["proj_err_row_px_p50"] == 1.5
    assert e["proj_err_col_beams_p50"] == 2.0


def test_merge_report_adds_target_distance_stats(tmp_path):
    """The distance axis of the E11 difficult-subset draft: per-frame
    |gantry head - calibrated P| via src_dir + gantry.csv re-read."""
    tool = _tool()
    src = _make_src(tmp_path)
    e = tool.inventory_recording(str(src / REC_ARC), REC_ARC)
    inv = {"src": str(src), "recordings": {REC_ARC: e}}
    e["src_dir"] = REC_ARC
    # synthetic arc: centre (1.2, 1.4), radius 1.08, head z = -0.25
    tool.merge_projection_report(inv, {REC_ARC: {
        "yaw_sign": 1, "P": [1.2, 1.4, -0.25]}})
    d = inv["recordings"][REC_ARC]
    assert 1.0 < d["dist_min_m"] < 1.2
    assert abs(d["dist_median_m"] - 1.08) < 0.05
    assert 1.0 < d["dist_max_m"] < 1.2
    # no P in the report -> no distance fields (old-report compatibility)
    e2 = tool.inventory_recording(str(src / REC_LINE), REC_LINE)
    inv2 = {"src": str(src), "recordings": {REC_LINE: e2}}
    e2["src_dir"] = REC_LINE
    tool.merge_projection_report(inv2, {REC_LINE: {"yaw_sign": 0}})
    assert "dist_median_m" not in inv2["recordings"][REC_LINE]


def test_merge_report_polar_head_keeps_path_geometry(tmp_path):
    tool = _tool()
    src = _make_src(tmp_path)
    inv = {"recordings": {
        REC_LINE: tool.inventory_recording(str(src / REC_LINE), REC_LINE)}}
    tool.merge_projection_report(inv, {REC_LINE: {"yaw_sign": 0}})
    e = inv["recordings"][REC_LINE]
    assert e["head"] == "constant" and e["geometry"] == "line"


def test_cli_writes_inventory_with_strata(tmp_path):
    import subprocess
    src = _make_src(tmp_path)
    out = tmp_path / "inventory.json"
    r = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "tools", "uxo_inventory.py"),
         "--src", str(src), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    inv = json.loads(out.read_text())
    assert set(inv["recordings"]) == {REC_ARC, REC_LINE}
    # stratification keyed category|geometry
    assert set(inv["stratification"]) == {"mortar_shell|arc",
                                          "mortar_shell|line"}
    assert inv["stratification"]["mortar_shell|arc"] == [REC_ARC]


def test_category_dir_audit_branches():
    tool = _tool()
    e = {"category": "mortar_shell"}
    tool.apply_category_dir_audit(e, None)                # flat layout
    assert (e["category_dir"], e["category_dir_matches_labels"]) == (None, None)
    tool.apply_category_dir_audit(e, "15cm_mortar")       # verified, match
    assert e["category_dir_matches_labels"] is True
    tool.apply_category_dir_audit(e, "100lbs_bomb")       # verified, mismatch
    assert e["category_dir_matches_labels"] is False
    tool.apply_category_dir_audit(e, "test_cylinder")     # unmapped: no claim
    assert e["category_dir_matches_labels"] is None
    # sonar-only recording (no labels -> no category) under a mapped
    # dir: indeterminate, NOT a mismatch (found in the 2026-11-02
    # real-data smoke: 2023-09-21_122115 under 20lbs_incendiary)
    unlabeled = {"category": None}
    tool.apply_category_dir_audit(unlabeled, "20lbs_incendiary")
    assert unlabeled["category_dir_matches_labels"] is None


def test_cli_class_grouped_layout(tmp_path):
    """The main package (data_export_recordings.7z, verified
    2026-11-02) nests recordings under category dirs; the CLI must
    inventory that layout and carry the dir as an audit field."""
    import shutil
    import subprocess
    src = _make_src(tmp_path)
    grouped = tmp_path / "grouped"
    cls = grouped / "15cm_mortar"
    cls.mkdir(parents=True)
    for rec in (REC_ARC, REC_LINE):
        shutil.move(str(src / rec), str(cls / rec))
    out = tmp_path / "inventory.json"
    r = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "tools", "uxo_inventory.py"),
         "--src", str(grouped), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    inv = json.loads(out.read_text())
    assert set(inv["recordings"]) == {REC_ARC, REC_LINE}
    for e in inv["recordings"].values():
        assert e["category"] == "mortar_shell"      # labels stay authoritative
        assert e["category_dir"] == "15cm_mortar"
        assert e["category_dir_matches_labels"] is True
    # strata stay keyed on the label-derived category
    assert set(inv["stratification"]) == {"mortar_shell|arc",
                                          "mortar_shell|line"}
