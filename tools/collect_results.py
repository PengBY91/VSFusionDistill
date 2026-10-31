#!/usr/bin/env python
"""Collect per-branch, per-level results into results.csv (F2 §6).

Walks work_dirs/*/ looking for ap_summary.json files (written by
tools/test.py) — one row per (run, branch, level) — and appends/merges
them into a single CSV keyed by run name.  Re-running merges by (run,
branch, level): last writer wins, so re-tested runs update in place.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIELDS = ["run", "branch", "level", "gt", "AP", "AP50", "AP75", "AP_small",
          "AP_medium", "AP_large", "AR@100"]


def find_summaries(work_dirs):
    for run in sorted(os.listdir(work_dirs)):
        run_dir = os.path.join(work_dirs, run)
        if not os.path.isdir(run_dir):
            continue
        summary = os.path.join(run_dir, "ap_summary.json")
        if os.path.exists(summary):
            yield run, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dirs")
    parser.add_argument("--out", required=True, help="results.csv path")
    args = parser.parse_args()

    rows = {}
    for run, summary in find_summaries(args.work_dirs):
        import json
        with open(summary) as f:
            for r in json.load(f):
                row = dict(run=run)
                # (0.0 is a valid AP; only a missing gt defaults to "all")
                row.update({k: (r[k] if k in r else
                                ("all" if k == "gt" else None))
                            for k in FIELDS if k != "run"})
                rows[(run, r["branch"], r["level"], row["gt"])] = row

    # merge with existing csv (same key -> overwrite)
    if os.path.exists(args.out):
        with open(args.out, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["run"], row["branch"], row["level"],
                       row.get("gt") or "all")
                row["gt"] = key[3]
                if key not in rows:
                    rows[key] = row

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
