#!/usr/bin/env python
"""F2 TODO 14.2 item 4: project UXO sonar boxes + cross-modal pi.

Calibrates each recording in a UXO paired-format root (camera GT
bearings + sonar blob echoes + gantry arc geometry), projects the
target into every sonar frame, then writes sonar.json annotations and
pairs.csv.  Also emits a calibration/projection-error report and an
optional contact sheet for eyeball checks.

    python tools/project_uxo_sonar.py --paired data/paired/uxo \
        --write --contact-sheet 12
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _contact_sheet(paired_root, entries, cam_bbox, dst, n=12):
    """Rows of [camera+GT | sonar+projected(red)+blob(green)]."""
    sample_ids = sorted(entries)
    step = max(1, len(sample_ids) // n)
    rows = []
    for sid in sample_ids[::step][:n]:
        e = entries[sid]
        cam = cv2.imread(os.path.join(paired_root, "images", "camera",
                                      f"{sid}.jpg"))
        son = cv2.imread(os.path.join(paired_root, "images", "sonar",
                                      f"{sid}.pgm"), cv2.IMREAD_UNCHANGED)
        if cam is None or son is None:
            continue
        son = cv2.applyColorMap(
            cv2.normalize(son, None, 0, 255, cv2.NORM_MINMAX),
            cv2.COLORMAP_TURBO)
        x, y, w, h = e["bbox"]
        cv2.rectangle(son, (int(x), int(y)), (int(x + w), int(y + h)),
                      (0, 0, 255), 1)
        r, c = e["projected"]
        cv2.drawMarker(son, (int(c), int(r)), (0, 0, 255), cv2.MARKER_CROSS,
                       7, 1)
        if e["blob"] is not None:
            br, bc, bw, bh = e["blob"]
            cv2.rectangle(son, (int(bc - bw / 2), int(br - bh / 2)),
                          (int(bc + bw / 2), int(br + bh / 2)),
                          (0, 255, 0), 1)
        if sid in cam_bbox:
            bx, by, bw2, bh2 = cam_bbox[sid]
            cv2.rectangle(cam, (int(bx), int(by)),
                          (int(bx + bw2), int(by + bh2)), (0, 0, 255), 4)
        rows.append(np.hstack([cv2.resize(cam, (640, 360)),
                               cv2.resize(son, (640, 360))]))
    if rows:
        cv2.imwrite(dst, np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"contact sheet -> {dst}")


def _calib_to_dict(calib) -> dict:
    """Serialise a CalibResult (numpy fields -> lists) for the cache."""
    return dict(recording=calib.recording, yaw_sign=calib.yaw_sign,
                circle_centre=calib.circle_centre.tolist(),
                phi_ref=calib.phi_ref, yaw_span_deg=calib.yaw_span_deg,
                R_cam0=calib.R_cam0.tolist(), R_son0=calib.R_son0.tolist(),
                P=calib.P.tolist(), b_rng=calib.b_rng,
                n_cam=calib.n_cam, n_sonar=calib.n_sonar,
                rms_cam_deg=calib.rms_cam_deg, rms_az_deg=calib.rms_az_deg,
                rms_rng_m=calib.rms_rng_m)


def _calib_from_dict(d: dict):
    from aodet.datasets.converters.uxo_projection import CalibResult
    return CalibResult(
        recording=d["recording"], yaw_sign=d["yaw_sign"],
        circle_centre=np.asarray(d["circle_centre"]), phi_ref=d["phi_ref"],
        yaw_span_deg=d["yaw_span_deg"], R_cam0=np.asarray(d["R_cam0"]),
        R_son0=np.asarray(d["R_son0"]), P=np.asarray(d["P"]),
        b_rng=d["b_rng"], n_cam=d["n_cam"], n_sonar=d["n_sonar"],
        rms_cam_deg=d["rms_cam_deg"], rms_az_deg=d["rms_az_deg"],
        rms_rng_m=d["rms_rng_m"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paired", default="data/paired/uxo")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--write", action="store_true",
                    help="write sonar.json + pairs.csv (default: dry run)")
    ap.add_argument("--contact-sheet", type=int, default=0,
                    help="write projection_check.jpg with N sampled rows")
    ap.add_argument("--use-cache", action="store_true",
                    help="load cached calibrations from projection_calib.json "
                         "instead of re-running calibrate() (~10 min per "
                         "recording); the cache is (re)written every run")
    args = ap.parse_args()

    from aodet.datasets.converters.uxo_projection import (
        CalibResult, anchor_b_rng, calibrate, load_observations,
        project_uxo_sonar)
    from aodet.datasets.paired_format import PairedData, write_paired

    with open(os.path.join(args.paired, "camera.json")) as f:
        camera = json.load(f)
    with open(os.path.join(args.paired, "frame_meta.csv"), newline="") as f:
        frame_meta = [dict(r) for r in csv.DictReader(f)]
    son_img_of_sample = {r["sample_id"]: int(r["sonar_image_id"])
                         for r in frame_meta}

    cache_path = os.path.join(args.paired, "projection_calib.json")
    cache = {}
    if args.use_cache and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        print(f"calibrations from cache -> {cache_path}")

    recs = load_observations(args.paired)
    report, all_entries, obs_by_sample = {}, {}, {}
    for rec in sorted(recs):
        obs = recs[rec]
        anchor = None
        if rec in cache:
            calib = _calib_from_dict(cache[rec])
        else:
            calib = calibrate(rec, obs, min_frames=args.min_frames)
            # E05 fix: move the range-offset the (P_radial, b_rng)
            # degeneracy leaked into P back into b_rng, anchored on the
            # bearing-gated blob frames (see anchor_b_rng docstring)
            calib, anchor = anchor_b_rng(calib, obs)
            cache[rec] = _calib_to_dict(calib)
        entries = project_uxo_sonar(args.paired, calib, obs)
        all_entries.update(entries)
        for o in obs:
            obs_by_sample[o.sample_id] = o
        dcols = [e["blob_delta"][0] for e in entries.values()
                 if e["blob_delta"]]
        drows = [e["blob_delta"][1] for e in entries.values()
                 if e["blob_delta"]]
        src = {s: sum(1 for e in entries.values() if e["source"] == s)
               for s in ("blob", "projected")}
        n_target = sum(1 for o in obs if o.category_id is not None)
        report[rec] = dict(
            anchor=anchor,
            yaw_sign=calib.yaw_sign,
            yaw_span_deg=round(calib.yaw_span_deg, 1),
            circle_centre=[round(float(v), 3) for v in calib.circle_centre],
            P=[round(float(v), 3) for v in calib.P],
            b_rng_m=round(calib.b_rng, 3),
            n_cam=calib.n_cam, n_sonar=calib.n_sonar,
            dropped_frames=n_target - len(entries),
            rms_cam_deg=round(calib.rms_cam_deg, 2),
            rms_az_deg=round(calib.rms_az_deg, 2),
            rms_rng_m=round(calib.rms_rng_m, 3),
            source_counts=src,
            proj_err_col_beams=[round(float(np.median(dcols)), 2),
                                round(float(np.percentile(dcols, 95)), 2)]
            if dcols else None,
            proj_err_row_px=[round(float(np.median(drows)), 1),
                             round(float(np.percentile(drows, 95)), 1)]
            if drows else None)
        r = report[rec]
        print(f"{rec}: yaw={r['yaw_sign']:+d} span={r['yaw_span_deg']}deg "
              f"cam={r['rms_cam_deg']}deg az={r['rms_az_deg']}deg "
              f"rng={r['rms_rng_m']}m b={r['b_rng_m']}m P={r['P']} "
              f"blob/proj/drop={src['blob']}/{src['projected']}/"
              f"{r['dropped_frames']} "
              f"dcol(p50,p95)={r['proj_err_col_beams']}")

    out_json = os.path.join(args.paired, "projection_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"report -> {out_json}")

    if not args.use_cache:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"calib cache ({len(cache)} recordings) -> {cache_path}")

    if args.contact_sheet:
        ann_of_img = {a["image_id"]: a["bbox"]
                      for a in camera["annotations"]}
        cam_bbox = {o.sample_id: ann_of_img[o.cam_ann_id]
                    for o in obs_by_sample.values() if o.cam_ann_id}
        _contact_sheet(args.paired, all_entries, cam_bbox,
                       os.path.join(args.paired, "projection_check.jpg"),
                       n=args.contact_sheet)

    if args.write:
        with open(os.path.join(args.paired, "sonar.json")) as f:
            sonar = json.load(f)
        sonar["annotations"] = []
        pairs = []
        for sid, e in sorted(all_entries.items()):
            sonar["annotations"].append({
                "id": len(pairs) + 1,
                "image_id": son_img_of_sample[sid],
                "category_id": e["category_id"],
                "bbox": [round(v, 2) for v in e["bbox"]],
                "area": round(e["bbox"][2] * e["bbox"][3], 2),
                "iscrowd": 0, "segmentation": [],
                "source": e["source"]})
            pairs.append({"camera_ann_id": e["cam_ann_id"],
                          "sonar_ann_id": len(pairs) + 1,
                          "sample_id": sid})
        write_paired(args.paired, PairedData(
            camera=camera, sonar=sonar, pairs=pairs,
            frame_meta=frame_meta, root=args.paired))
        print(f"wrote {len(pairs)} sonar annotations + pairs -> "
              f"{args.paired}")


if __name__ == "__main__":
    main()
