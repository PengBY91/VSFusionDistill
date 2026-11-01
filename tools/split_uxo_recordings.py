#!/usr/bin/env python
"""E11 prereq: stratified recording-level split for the full UXO set.

Consumes tools/uxo_inventory.py output (with calibration residuals
merged from projection_report.json) and emits the split manifest
tools/uxo_to_paired.py --splits-json consumes.  Implements the frozen
E11 划分规则 (doc/experiments.md E11, 出数前冻结):

1. 标定质量过滤先于划分、与模型无关: exclude a recording when
   blob frames < 30, or anchored row-residual p50 > 10 px, or bearing
   residual rms > 3 deg.  Calibration numbers must be present for
   every recording (the filter is defined on them); recordings missing
   them abort the run -- calibrate first.
2. Recording-level split, stratified by category x geometry
   (arc/line/aim), ratio ~6:2:2 (train/val/test); per category at
   least 3 recordings in val and in test, at least 6 in train.
   Infeasible categories abort with a per-category shortfall list
   (the frozen fallback is 3-fold cross-validation, a human decision
   to record before launch, not something this tool does silently).
3. The five sample recordings were used as a development set: they go
   to train or val only, never test.
4. Moving-target recordings are assigned normally (they get their own
   row in every table, they are not excluded).
5. Split seed fixed at 0.  The manifest sha256 is recorded in the
   experiment ledger before training starts.

    python tools/split_uxo_recordings.py \
        --inventory data/raw/uxo-full/inventory.json \
        --out data/raw/uxo-full/split_manifest.json
"""

import argparse
import hashlib
import json
import random
import sys

# the sample.7z five (E11 划分规则): used as a development set through
# E10, never eligible for test
DEV_RECORDINGS = (
    "2023-09-20_180722",
    "2023-09-21_100927",
    "2023-09-21_105856",
    "2023-09-21_123500",
    "2023-09-21_124735",
)

MIN_BLOB_FRAMES = 30
MAX_ROW_RESID_PX = 10.0
MAX_AZ_RMS_DEG = 3.0
MIN_PER_CLASS_VAL = 3
MIN_PER_CLASS_TEST = 3
MIN_PER_CLASS_TRAIN = 6
VAL_FRAC = 0.2
TEST_FRAC = 0.2


def calibration_exclusion(e: dict):
    """None when the recording passes the 标定质量过滤, else the reason
    string (frozen thresholds)."""
    blob = e.get("n_blob_frames")
    row = e.get("proj_err_row_px_p50")
    az = e.get("rms_az_deg")
    missing = [k for k, v in (("n_blob_frames", blob),
                              ("proj_err_row_px_p50", row),
                              ("rms_az_deg", az)) if v is None]
    if missing:
        return f"missing calibration numbers: {', '.join(missing)}"
    if blob < MIN_BLOB_FRAMES:
        return f"blob frames {blob} < {MIN_BLOB_FRAMES}"
    if row > MAX_ROW_RESID_PX:
        return f"row residual p50 {row} px > {MAX_ROW_RESID_PX}"
    if az > MAX_AZ_RMS_DEG:
        return f"bearing residual rms {az} deg > {MAX_AZ_RMS_DEG}"
    return None


def assign(inventory: dict, seed: int = 0):
    """-> (manifest recordings dict, infeasibility list).  Within a
    category, candidates are drawn round-robin over geometry groups so
    every split samples all geometries; test is filled first (it has
    the extra dev-set restriction), then val, the rest is train."""
    rng = random.Random(seed)
    recs = inventory["recordings"]
    manifest, infeasible = {}, []

    by_cat = {}
    for name, e in sorted(recs.items()):
        reason = calibration_exclusion(e)
        manifest[name] = {
            "split": "excluded" if reason else None,
            "exclude_reason": reason or "",
            "category": e.get("category"),
            "geometry": e.get("geometry"),
            "moving_target": bool(e.get("moving_target")),
            "n_blob_frames": e.get("n_blob_frames"),
            "proj_err_row_px_p50": e.get("proj_err_row_px_p50"),
            "rms_az_deg": e.get("rms_az_deg"),
            "dev_recording": name in DEV_RECORDINGS,
        }
        if not reason:
            by_cat.setdefault(e["category"], []).append(name)

    for cat, names in sorted(by_cat.items()):
        groups = {}
        for n in names:
            groups.setdefault(recs[n].get("geometry"), []).append(n)
        for g in groups.values():
            rng.shuffle(g)
        order, keys = [], sorted(groups)
        while any(groups[k] for k in keys):
            for k in keys:                     # geometry round-robin
                if groups[k]:
                    order.append(groups[k].pop())
        n = len(order)
        q_test = max(MIN_PER_CLASS_TEST, round(TEST_FRAC * n))
        q_val = max(MIN_PER_CLASS_VAL, round(VAL_FRAC * n))
        if n < q_test + q_val + MIN_PER_CLASS_TRAIN:
            infeasible.append(
                f"{cat}: {n} usable recordings < {q_test} test + "
                f"{q_val} val + {MIN_PER_CLASS_TRAIN} train")
            continue
        test, val, train = [], [], []
        it = iter(order)
        for name in it:                        # test: no dev recordings
            if len(test) == q_test:
                break
            if manifest[name]["dev_recording"]:
                val.append(name)               # parked, counted below
                continue
            test.append(name)
        else:
            name = None
        rest = [name] + list(it) if name else list(it)
        for name in rest:
            if len(val) < q_val:
                val.append(name)
            else:
                train.append(name)
        if len(test) < q_test:                 # dev recordings ate val
            infeasible.append(
                f"{cat}: only {len(test)} test-eligible recordings "
                f"(dev recordings are test-excluded), need {q_test}")
            continue
        if len(val) < q_val or len(train) < MIN_PER_CLASS_TRAIN:
            infeasible.append(
                f"{cat}: short after allocation "
                f"(val {len(val)}/{q_val}, train {len(train)}/"
                f"{MIN_PER_CLASS_TRAIN})")
            continue
        for name in test:
            manifest[name]["split"] = "test"
        for name in val:
            manifest[name]["split"] = "val"
        for name in train:
            manifest[name]["split"] = "train"
    return manifest, infeasible


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inventory", required=True,
                    help="inventory json from tools/uxo_inventory.py "
                         "(with calibration residuals merged)")
    ap.add_argument("--out", required=True, help="split manifest json")
    ap.add_argument("--seed", type=int, default=0,
                    help="split seed (frozen rule: 0)")
    args = ap.parse_args()

    with open(args.inventory, encoding="utf-8") as f:
        inventory = json.load(f)
    manifest_recs, infeasible = assign(inventory, seed=args.seed)
    if infeasible:
        print("INFEASIBLE under the frozen constraints:", file=sys.stderr)
        for line in infeasible:
            print(f"  {line}", file=sys.stderr)
        print("frozen fallback is recording-level 3-fold CV -- record "
              "the switch in the experiment ledger before launch",
              file=sys.stderr)
        sys.exit(2)

    counts = {}
    for m in manifest_recs.values():
        counts[m["split"]] = counts.get(m["split"], 0) + 1
    manifest = {"seed": args.seed, "counts": counts,
                "recordings": manifest_recs}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    print(json.dumps(counts, indent=2))
    print(f"manifest -> {args.out}")
    print(f"sha256 {digest}")       # goes into the ledger before launch


if __name__ == "__main__":
    main()
