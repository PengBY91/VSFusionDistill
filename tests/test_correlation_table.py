"""End-to-end test of tools/correlation_table.py on a fake run dir.

Frames 1-4 (split test) share one GT box per side; frames 5-8 are train
frames with GT but no predictions — they must not enter the pairs
(review 2026-10-31 §4.1: they used to count as forced double misses).
d0: frames 1,2 missed by BOTH branches (phi = +1), 3,4 hit by both.  d1: camera silent everywhere
(D-inf style collapse).  d2: camera's frame-4 hit is below tau=0.25,
so it counts as a miss while sonar hits.
"""

import csv
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _merge_gt(ids):
    gt = {"images": [], "annotations": [],
          "categories": [{"id": 3, "name": "mortar"}]}
    for i in ids:
        gt["images"].append({"id": i, "width": 128, "height": 441})
        gt["annotations"].append(
            {"id": i, "image_id": i, "category_id": 3,
             "bbox": [40, 100, 20, 20], "area": 400, "iscrowd": 0})
    return gt


def _build(tmp_path):
    paired = tmp_path / "paired"
    paired.mkdir()
    json.dump(_merge_gt(range(1, 9)), open(paired / "camera.json", "w"))
    json.dump(_merge_gt(range(1, 9)), open(paired / "sonar.json", "w"))
    with open(paired / "frame_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "camera_image_id", "sonar_image_id",
                    "split", "recording"])
        for i in range(1, 9):
            w.writerow([f"s{i}", i, i, "test" if i <= 4 else "train",
                        "recA" if i <= 2 else "recB"])

    run = tmp_path / "run"
    run.mkdir()
    hit = {"category_id": 3, "bbox": [40, 100, 20, 20], "score": 0.9}
    low = {"category_id": 3, "bbox": [40, 100, 20, 20], "score": 0.1}
    json.dump([dict(hit, image_id=i) for i in (3, 4)],
              open(run / "pred.camera.d0.json", "w"))
    json.dump([dict(hit, image_id=i) for i in (3, 4)],
              open(run / "pred.sonar.d0.json", "w"))
    json.dump([], open(run / "pred.camera.d1.json", "w"))
    json.dump([dict(hit, image_id=i) for i in (3, 4)],
              open(run / "pred.sonar.d1.json", "w"))
    json.dump([dict(hit, image_id=3), dict(low, image_id=4)],
              open(run / "pred.camera.d2.json", "w"))
    json.dump([dict(hit, image_id=i) for i in (3, 4)],
              open(run / "pred.sonar.d2.json", "w"))
    return run, paired


def test_correlation_table(tmp_path):
    run, paired = _build(tmp_path)
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "correlation_table.py"),
         str(run), "--paired", str(paired),
         "--levels", "d0", "d1", "d2"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    out = os.path.join(run, "correlation.csv")
    rows = {row["level"]: row for row in csv.DictReader(open(out))}
    assert len(rows) == 3
    assert int(rows["d0"]["n_pairs"]) == 4
    assert abs(float(rows["d0"]["phi"]) - 1.0) < 1e-6      # co-missing
    assert abs(float(rows["d1"]["p_miss_camera"]) - 1.0) < 1e-6
    # camera misses 1,2 (silent) and 4 (score 0.1 < tau); sonar only 1,2
    assert abs(float(rows["d2"]["p_miss_camera"]) - 0.75) < 1e-6
    assert abs(float(rows["d2"]["p_miss_sonar"]) - 0.50) < 1e-6
    # lift = P(sonar miss | camera miss) - P(sonar miss) = 2/3 - 1/2
    assert abs(float(rows["d2"]["delta_p_miss"]) - (2/3 - 0.5)) < 1e-6


def test_correlation_table_reports_both_sonar_heads_and_filters(tmp_path):
    run, paired = _build(tmp_path)
    hit = {"category_id": 3, "bbox": [40, 100, 20, 20], "score": 0.9}
    json.dump([dict(hit, image_id=i) for i in (1, 2, 3, 4)],
              open(run / "pred.sonar_uncoupled.d0.json", "w"))
    # projected sonar GT on frame 1 is dropped under the default blob view
    sonar = json.load(open(paired / "sonar.json"))
    sonar["annotations"][0]["source"] = "projected"
    json.dump(sonar, open(paired / "sonar.json", "w"))
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "correlation_table.py"),
         str(run), "--paired", str(paired), "--levels", "d0"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = {row["sonar_branch"]: row for row in
            csv.DictReader(open(os.path.join(run, "correlation.csv")))}
    assert set(rows) == {"sonar", "sonar_uncoupled"}
    assert all(int(row["n_pairs"]) == 3 for row in rows.values())
    assert float(rows["sonar_uncoupled"]["p_miss_sonar"]) == 0.0
    assert all(row["tau_source"] == "fixed" for row in rows.values())
    # recording filter (LOO-CV folds)
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "correlation_table.py"),
         str(run), "--paired", str(paired), "--levels", "d0",
         "--split", "all", "--recordings", "recB",
         "--sonar-gt-sources", "blob", "projected"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    rows = list(csv.DictReader(open(os.path.join(run, "correlation.csv"))))
    assert all(int(row["n_pairs"]) == 6 for row in rows)   # 3,4 + 5..8


def test_correlation_table_unfiltered_semantics(tmp_path):
    run, paired = _build(tmp_path)
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools", "correlation_table.py"),
         str(run), "--paired", str(paired), "--levels", "d0",
         "--split", "all", "--sonar-gt-sources", "blob", "projected"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    out = os.path.join(run, "correlation.csv")
    rows = {row["level"]: row for row in csv.DictReader(open(out))}
    # frames 1-8 all pair: projected GT counts, train frames co-miss
    assert int(rows["d0"]["n_pairs"]) == 8
