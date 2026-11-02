#!/usr/bin/env python
"""E11 prereq 4: per-recording inventory of a UXO source tree.

One row per recording: category (from labels), geometry (arc vs line
from the gantry path, plus the fitted head convention when a
projection report is supplied), annotated-frame count, blob-frame
count (the 标定质量过滤 input), calibration residuals (from
projection_report.json, produced by tools/project_uxo_sonar.py), and
the moving-target flag (rethink 2026-11-01 划分规则: 相机 GT 中心或声呐
GT 测距轴在录音内单调漂移超过画面 10% -- target motion gets its own row
in every E11 table, it is not a reason to exclude).

    python tools/uxo_inventory.py --src data/raw/uxo-full/extracted/... \
        --out data/raw/uxo-full/inventory.json \
        [--projection-report data/paired/uxo/projection_report.json]

Feeds tools/split_uxo_recordings.py (E11 prereq: the split stratifies
on category x geometry and filters on the calibration residuals this
tool carries).  Blob detection is the slow part (~10 ms/frame); the
calibration itself stays in the projection tool.
"""

import argparse
import csv
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from aodet.datasets.converters.uxo_projection import (  # noqa: E402
    _YAW_AIM, detect_blobs, fit_circle)
from aodet.datasets.converters.uxo_layout import (  # noqa: E402
    CATEGORY_DIR_TO_LABEL_CLASS, discover_recordings)

_FRAME_RE = re.compile(r"^\d{4}$")
_MONO_FRAC = 0.9    # sign consistency for "monotonic drift"
_DRIFT_FRAC = 0.10  # drift threshold: 10% of the image extent
_HEAD_OF_YAW = {0: "constant", 1: "polar", -1: "polar", _YAW_AIM: "aim"}


def _monotone_drift(vals) -> bool:
    """True when the sequence drifts one-sidedly far enough to be
    called target motion: sign sequence >= _MONO_FRAC one-sided AND
    total excursion > _DRIFT_FRAC * extent (extent = the axis this is
    called with, already normalised by the caller)."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 10:
        return False        # too short to call monotone drift
    d = np.diff(v)
    d = d[d != 0]
    if len(d) == 0:
        return False
    one_sided = abs(np.sign(d).sum()) / len(d) >= _MONO_FRAC
    return bool(one_sided and v.max() - v.min() > _DRIFT_FRAC)


def _read_csv_rows(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = "aris_frame_idx" if "aris_frame_idx" in row else "FrameIndex"
            try:
                out[int(row[key])] = row
            except (KeyError, ValueError):
                continue
    return out


def apply_category_dir_audit(entry: dict, group) -> None:
    """Cross-check upstream's category dir (data_export_recordings.7z
    groups recordings as <src>/<category_dir>/<rec>/) against the
    label-derived category, which stays authoritative -- the dir and
    label namespaces differ (15cm_mortar holds mortar_shell frames).
    ``category_dir_matches_labels`` is None for the flat layout, for
    dirs not in the verified map (no claim made, e.g. test_cylinder),
    and for recordings without labels (no category to check -- a
    sonar-only recording under 20lbs_incendiary is not a mismatch)."""
    entry["category_dir"] = group
    expected = CATEGORY_DIR_TO_LABEL_CLASS.get(group) if group else None
    entry["category_dir_matches_labels"] = (
        None if expected is None or entry.get("category") is None
        else entry["category"] == expected)


def inventory_recording(rec_dir: str, name: str) -> dict:
    """Inventory one recording directory (see module docstring)."""
    def _frames(sub, ext):
        d = os.path.join(rec_dir, sub)
        if not os.path.isdir(d):
            return set()
        return {f[:-len(ext) - 1] for f in os.listdir(d)
                if f.endswith("." + ext) and _FRAME_RE.match(
                    f[:-len(ext) - 1])}

    gopro = _frames("gopro", "jpg")
    labels = _frames("labels", "json")
    aris = _frames("aris_raw", "pgm")
    meta = _read_csv_rows(os.path.join(rec_dir, "aris_frame_meta.csv"))
    gantry = _read_csv_rows(os.path.join(rec_dir, "gantry.csv"))

    entry = {"recording": name, "n_annotated": len(labels),
             "n_gopro": len(gopro), "n_aris": len(aris),
             "annotated_with_gopro_and_aris":
                 len(labels & gopro & aris)}

    # category: one class per recording by design; a mix is a flag,
    # not an error
    classes = {}
    centers_u, centers_v = [], []
    aspects = []
    for idx in sorted(labels):
        try:
            with open(os.path.join(rec_dir, "labels", idx + ".json"),
                      encoding="utf-8") as f:
                lab = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        classes[lab.get("class", "?")] = \
            classes.get(lab.get("class", "?"), 0) + 1
        centers_u.append((lab["x_min"] + lab["x_max"]) / 2 * 3)  # 640x360
        centers_v.append((lab["y_min"] + lab["y_max"]) / 2 * 3)  # -> 1920x1080
        h = lab["y_max"] - lab["y_min"]
        if h > 0:                                  # GT aspect: pose proxy
            aspects.append((lab["x_max"] - lab["x_min"]) / h)
    entry["categories"] = classes
    entry["category"] = max(classes, key=classes.get) if classes else None
    entry["mixed_classes"] = len(classes) > 1

    # geometry from the gantry path (same heuristic as calibrate():
    # a Kasa fit through a noisy straight path returns a bogus circle,
    # so compare circle vs line residuals)
    G = np.array([[float(gantry[i]["x"]), float(gantry[i]["y"])]
                  for i in sorted(gantry)])
    if len(G) >= 10:
        _, circ_resid = fit_circle(G[:, 0], G[:, 1])
        xy = G - G.mean(axis=0)
        line_resid = float(np.linalg.svd(xy, compute_uv=False)[1]
                           / np.sqrt(len(xy)))
        entry["path"] = "arc" if circ_resid < 0.3 * line_resid else "line"
    else:
        entry["path"] = None
    entry["head"] = None            # filled from the projection report
    entry["geometry"] = entry["path"]  # refined once head is known

    # blob frames + sonar range-axis motion (over annotated frames --
    # the frames that produce GT; the 标定质量过滤 counts these)
    import cv2
    n_blob = 0
    blob_rows = []
    contrasts = []
    for idx in sorted(labels & aris):
        img = cv2.imread(os.path.join(rec_dir, "aris_raw", idx + ".pgm"),
                         cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        blobs = detect_blobs(img)
        if blobs:
            n_blob += 1
            blob_rows.append(blobs[0][1])   # row centre
            med = float(np.median(img))     # echo contrast: peak/bg
            if med > 0:
                contrasts.append(float(blobs[0][0]) / med)
    entry["n_blob_frames"] = n_blob
    entry["blob_ratio"] = round(n_blob / max(len(labels & aris), 1), 3)
    # E11 difficult-subset draft axes (rethink 9.2b): model-free,
    # recording-level clearing attributes
    entry["blob_contrast_median"] = (
        round(float(np.median(contrasts)), 3) if contrasts else None)
    entry["gt_aspect_median"] = (
        round(float(np.median(aspects)), 3) if aspects else None)
    n_rows = 441
    try:
        import yaml
        with open(os.path.join(rec_dir, "aris_file_meta.yaml"),
                  encoding="utf-8") as f:
            n_rows = int((yaml.safe_load(f) or {}).get(
                "SamplesPerChannel", 441))
    except Exception:
        pass
    # sonar moving: blob row centre drifting monotonically over >10% of
    # the range axis (画面 10%)
    entry["moving_target_sonar"] = (
        _monotone_drift([r / n_rows for r in blob_rows])
        if blob_rows else None)
    # camera moving: GT centre drifting monotonically over >10% of the
    # image width/height
    entry["moving_target_camera"] = (
        (_monotone_drift([u / 1920.0 for u in centers_u])
         or _monotone_drift([v / 1080.0 for v in centers_v]))
        if centers_u else None)
    entry["moving_target"] = bool(
        entry["moving_target_sonar"] or entry["moving_target_camera"])
    return entry


def merge_projection_report(inventory: dict, report: dict) -> int:
    """Fold calibration numbers into inventory entries; returns how
    many recordings were enriched.  The residuals drive the E11
    标定质量过滤 (blob 帧 < 30 / 行残差 p50 > 10 px / 方位残差 rms > 3°)."""
    n = 0
    recs = inventory["recordings"]
    for rec, r in report.items():
        if rec not in recs:
            continue
        e = recs[rec]
        e["head"] = _HEAD_OF_YAW.get(r.get("yaw_sign"))
        if e["path"]:
            # geometry for stratification: aim mode is its own stratum
            # (rethink 划分规则: 类别 × 几何（弧线/直线/瞄准模式）)
            e["geometry"] = "aim" if e["head"] == "aim" else e["path"]
        e["rms_cam_deg"] = r.get("rms_cam_deg")
        e["rms_az_deg"] = r.get("rms_az_deg")
        e["rms_rng_m"] = r.get("rms_rng_m")
        e["proj_err_row_px_p50"] = (r.get("proj_err_row_px") or [None])[0]
        e["proj_err_col_beams_p50"] = (r.get("proj_err_col_beams")
                                       or [None])[0]
        # sonar-target distance stats (E11 difficult-subset draft axis):
        # per-frame |gantry head position - calibrated target P|; the
        # head sweeps an arc, so min/median/max describe the run
        P = r.get("P")
        src_dir = e.get("src_dir")
        if P and src_dir:
            gantry = _read_csv_rows(os.path.join(
                inventory["src"], src_dir, "gantry.csv"))
            dists = []
            for i in sorted(gantry):
                g = gantry[i]
                try:
                    dists.append(math.dist(
                        (float(g["x"]), float(g["y"]), float(g["z"])),
                        (float(P[0]), float(P[1]), float(P[2]))))
                except (KeyError, ValueError, TypeError, IndexError):
                    continue
            if dists:
                e["dist_min_m"] = round(min(dists), 3)
                e["dist_median_m"] = round(float(np.median(dists)), 3)
                e["dist_max_m"] = round(max(dists), 3)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True,
                    help="dir containing the recording dirs")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--projection-report", default=None,
                    help="projection_report.json from "
                         "tools/project_uxo_sonar.py; adds calibration "
                         "residuals + the fitted head convention")
    args = ap.parse_args()

    recs = discover_recordings(args.src)
    if not recs:
        sys.exit(f"no recording dirs (YYYY-MM-DD_HHMMSS) under {args.src}")
    inventory = {"src": os.path.abspath(args.src), "recordings": {}}
    for rec, rec_dir, group in recs:
        inventory["recordings"][rec] = inventory_recording(rec_dir, rec)
        e = inventory["recordings"][rec]
        e["src_dir"] = os.path.relpath(rec_dir, args.src)
        apply_category_dir_audit(e, group)
        flag = ("" if e["category_dir_matches_labels"] in (None, True)
                else "  LABEL/DIR-MISMATCH")
        print(f"{rec}: cat={e['category']} dir={group or '-'} "
              f"path={e['path']} ann={e['n_annotated']} "
              f"blob={e['n_blob_frames']} moving={e['moving_target']}"
              f"{flag}")

    if args.projection_report:
        with open(args.projection_report, encoding="utf-8") as f:
            report = json.load(f)
        n = merge_projection_report(inventory, report)
        print(f"merged calibration residuals for {n} recordings")

    strat = {}
    for e in inventory["recordings"].values():
        key = (e["category"], e["geometry"])
        strat.setdefault(key, []).append(e["recording"])
    inventory["stratification"] = {
        f"{cat}|{geo}": sorted(v) for (cat, geo), v in sorted(
            strat.items(), key=lambda kv: str(kv[0]))}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    print(f"inventory ({len(inventory['recordings'])} recordings, "
          f"{len(inventory['stratification'])} strata) -> {args.out}")


if __name__ == "__main__":
    main()
