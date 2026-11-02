#!/usr/bin/env python
"""E11 option-C diagnosis (timeboxed): where do the row residuals that
broke the data gate come from?

Three suspect mechanisms (doc/logs/2026-11-02-e11-data-gate-infeasible.md):
1. b_rng pinned at the +-B_RNG_MAX clamp (range-offset degeneracy),
2. arc-endpoint frames (E03: 10-30 beam offsets at the ends of arc
   runs -- does the middle of the trajectory agree?),
3. degenerate target positions on low-sweep recordings (124735 et al).

Read-only: recomputes per-frame |blob_row - projected_row| from the
paired dir + cached calibrations (no refit), writes a per-recording
summary json + per-frame csv for trajectory analysis.  row_p50_all
must reproduce inventory's proj_err_row_px_p50 (same calib, same
frames); a mismatch means this tool is measuring something else.

    python tools/diag_row_residual.py --paired <paired_root> \
        --cache <cache.json> --inventory <inventory.json> \
        --out-summary <summary.json> --out-frames <frames.csv>
"""

import argparse
import csv
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from aodet.datasets.converters.uxo_projection import (  # noqa: E402
    _B_RNG_MAX, _GATE_COL, _GATE_RNG, load_observations, predict)

_spec = importlib.util.spec_from_file_location(
    "project_uxo_sonar_tool",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "project_uxo_sonar.py"))
_proj_tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_proj_tool)
_calib_from_dict = _proj_tool._calib_from_dict

MID_LO, MID_HI = 0.2, 0.8        # "middle 60%" of the temporal order


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paired", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-frames", required=True)
    args = ap.parse_args()

    inv = json.load(open(args.inventory, encoding="utf-8"))["recordings"]
    cache = json.load(open(args.cache, encoding="utf-8"))["recordings"]

    t0 = time.monotonic()
    print(f"[0s] loading observations (blob detection on every frame of "
          f"every recording)...", flush=True)
    recs = load_observations(args.paired)
    print(f"[{time.monotonic()-t0:.0f}s] observations loaded, projecting...",
          flush=True)

    summary, frames = {}, []
    for rec in sorted(recs):
        if rec not in cache:
            continue                       # no calibration -> skip
        e_inv = inv.get(rec, {})
        calib = _calib_from_dict(cache[rec]["calib"])
        target = sorted((o for o in recs[rec]
                         if o.category_id is not None and o.blob is not None),
                        key=lambda o: o.frame_idx)
        rows = []                          # (t, drow, dcol, gated)
        for t_i, o in enumerate(target):
            row_p, col_p = predict(calib, o)
            br, bc = o.blob[0], o.blob[1]
            rng = float(np.linalg.norm(calib.P - np.asarray(o.gantry)))
            gated = (abs(bc - col_p) <= _GATE_COL
                     and abs(o.rng - (rng + calib.b_rng)) <= _GATE_RNG)
            rows.append((t_i / max(len(target) - 1, 1), abs(br - row_p),
                         abs(bc - col_p), gated))

        drows = np.array([r[1] for r in rows]) if rows else np.array([])
        mid = np.array([MID_LO <= r[0] <= MID_HI for r in rows]) \
            if rows else np.array([], dtype=bool)

        def p50(a):
            return round(float(np.median(a)), 1) if len(a) else None

        inv_p50 = e_inv.get("proj_err_row_px_p50")
        summary[rec] = {
            "category": e_inv.get("category"),
            "geometry": e_inv.get("geometry"),
            "row_p50_all": p50(drows),
            "row_p50_mid60": p50(drows[mid]) if rows else None,
            "row_p50_ends": p50(drows[~mid]) if rows else None,
            "n_frames_blob": len(rows),
            "gated_frac": (round(float(np.mean([r[3] for r in rows])), 3)
                           if rows else None),
            "b_rng": round(float(calib.b_rng), 3),
            "b_clamped": abs(abs(float(calib.b_rng)) - _B_RNG_MAX) < 1e-6,
            "P": [round(float(v), 3) for v in calib.P],
            "yaw_span_deg": round(float(calib.yaw_span_deg), 1),
            "rms_az_deg": e_inv.get("rms_az_deg"),
            "inv_row_p50": inv_p50,
            "reproduces_inventory": (
                None if inv_p50 is None or not len(rows)
                else abs(p50(drows) - inv_p50) <= 0.15),
            "passes_gate": (
                e_inv.get("n_blob_frames", 0) >= 30
                and (inv_p50 if inv_p50 is not None else 1e9) <= 10.0
                and (e_inv.get("rms_az_deg") or 1e9) <= 3.0),
        }
        frames.extend(dict(recording=rec, t=round(r[0], 4),
                           drow=round(r[1], 2), dcol=round(r[2], 2),
                           gated=bool(r[3])) for r in rows)
        s = summary[rec]
        print(f"{rec}: n={len(rows)} p50all={s['row_p50_all']} "
              f"p50mid={s['row_p50_mid60']} p50ends={s['row_p50_ends']} "
              f"gate={s['gated_frac']} b={s['b_rng']}"
              f"{' CLAMP' if s['b_clamped'] else ''} "
              f"span={s['yaw_span_deg']}", flush=True)

    with open(args.out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    with open(args.out_frames, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["recording", "t", "drow",
                                          "dcol", "gated"])
        w.writeheader()
        w.writerows(frames)
    n_inv_ok = sum(1 for s in summary.values()
                   if s["reproduces_inventory"] is True)
    print(f"summary -> {args.out_summary} ({len(summary)} recordings, "
          f"{n_inv_ok} reproduce inventory row_p50)")
    print(f"frames  -> {args.out_frames} ({len(frames)} rows)")


if __name__ == "__main__":
    main()
