"""Known-answer tests for the class-transplant upper bound
(aodet.evaluation.class_transplant, tools/class_transplant.py).

Fixture: six val frames, one target each; sonar GT classes 1,1,2,2,3,3.
Camera image ids are sonar id + 100, so pairing must go through pi
(frame_meta), never through equal ids.  Sonar predictions localise every
target perfectly but call everything class 1, so class-aware AP50 of the
original labels is 1/3 (class 1 perfect, classes 2 and 3 zero) while
class-agnostic AP50 is 1.0.
"""

import csv
import json
import os
import subprocess
import sys

import pytest

pytest.importorskip("pycocotools")

from aodet.evaluation.class_transplant import (  # noqa: E402
    agnostic_nms, camera_class_dist, evaluate_transplant)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = [{"id": 1, "name": "bomb"}, {"id": 2, "name": "mortar"},
        {"id": 3, "name": "grenade"}]
GT_CLS = {1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3}
SBOX = [40, 100, 20, 20]
CBOX = [500, 300, 200, 120]


def _gt(offset, box):
    g = {"images": [], "annotations": [], "categories": CATS}
    for s, c in GT_CLS.items():
        i = s + offset
        g["images"].append({"id": i, "width": 1920, "height": 1080})
        g["annotations"].append({"id": i, "image_id": i, "category_id": c,
                                 "bbox": box, "area": box[2] * box[3],
                                 "iscrowd": 0, "source": "blob"})
    return g


SONAR_GT = _gt(0, SBOX)
CAMERA_GT = _gt(100, CBOX)
PAIRS = [{"sonar_image_id": s, "camera_image_id": s + 100} for s in GT_CLS]
SONAR_ALL_CLS1 = [{"image_id": s, "category_id": 1, "bbox": SBOX,
                   "score": 0.9} for s in GT_CLS]


def _cam(cls_of):
    return [{"image_id": s + 100, "category_id": c, "bbox": CBOX,
             "score": 0.8} for s, c in cls_of.items()]


def _run(camera_preds, sonar_preds=SONAR_ALL_CLS1, **kw):
    return evaluate_transplant(SONAR_GT, CAMERA_GT, sonar_preds,
                               camera_preds, PAIRS, list(GT_CLS), **kw)


def test_perfect_camera_lifts_to_ceiling():
    r = _run(_cam(GT_CLS))
    p = r["pooled"]
    assert p["original"]["AP50"] == pytest.approx(1 / 3, abs=1e-6)
    assert p["class_agnostic"]["AP50"] == pytest.approx(1.0)
    for v in ("camera_hard", "camera_soft", "oracle"):
        assert p[v]["AP50"] == pytest.approx(1.0), v
    assert r["delta_vs_original"]["camera_hard"]["AP50"] == \
        pytest.approx(2 / 3, abs=1e-6)
    assert r["diagnostics"]["camera_top1_acc_on_gt_images"] == 1.0
    assert r["diagnostics"]["n_fallback_images"] == 0


def test_silent_camera_falls_back_to_sonar_labels():
    r = _run([])
    p = r["pooled"]
    assert p["camera_hard"]["AP50"] == pytest.approx(p["original"]["AP50"])
    assert p["camera_soft"]["AP50"] == pytest.approx(p["original"]["AP50"])
    assert p["oracle"]["AP50"] == pytest.approx(1.0)
    assert r["diagnostics"]["n_fallback_images"] == 6
    assert r["diagnostics"]["camera_top1_acc_on_gt_images"] is None


def test_all_wrong_camera_scores_zero_oracle_unaffected():
    wrong = {s: c % 3 + 1 for s, c in GT_CLS.items()}
    r = _run(_cam(wrong))
    assert r["pooled"]["camera_hard"]["AP50"] == pytest.approx(0.0)
    assert r["pooled"]["oracle"]["AP50"] == pytest.approx(1.0)
    assert r["diagnostics"]["camera_top1_acc_on_gt_images"] == 0.0


def test_pairing_goes_through_pi_not_equal_ids():
    # camera dets placed on sonar ids (no offset) must not be picked up
    misplaced = [dict(d, image_id=d["image_id"] - 100)
                 for d in _cam(GT_CLS)]
    r = _run(misplaced)
    assert r["diagnostics"]["n_fallback_images"] == 6
    assert r["pooled"]["camera_hard"]["AP50"] == \
        pytest.approx(r["pooled"]["original"]["AP50"])


def test_soft_keeps_ranking_when_right_class_dominates():
    cam = []
    for s, c in GT_CLS.items():
        cam.append({"image_id": s + 100, "category_id": c, "bbox": CBOX,
                    "score": 0.8})
        cam.append({"image_id": s + 100, "category_id": c % 3 + 1,
                    "bbox": CBOX, "score": 0.2})
    r = _run(cam)
    assert r["pooled"]["camera_soft"]["AP50"] == pytest.approx(1.0)
    assert r["pooled"]["camera_hard"]["AP50"] == pytest.approx(1.0)


def test_category_mapping_by_name():
    # camera json numbers the same classes differently
    cam_gt = dict(CAMERA_GT, categories=[{"id": 11, "name": "bomb"},
                                         {"id": 12, "name": "mortar"},
                                         {"id": 13, "name": "grenade"}])
    cam = [dict(d, category_id=d["category_id"] + 10) for d in _cam(GT_CLS)]
    r = evaluate_transplant(SONAR_GT, cam_gt, SONAR_ALL_CLS1, cam, PAIRS,
                            list(GT_CLS))
    assert r["pooled"]["camera_hard"]["AP50"] == pytest.approx(1.0)


def test_agnostic_nms_collapses_cross_class_duplicates():
    dets = [{"image_id": 1, "category_id": 1, "bbox": SBOX, "score": 0.9},
            {"image_id": 1, "category_id": 2, "bbox": SBOX, "score": 0.5},
            {"image_id": 1, "category_id": 2, "bbox": [0, 0, 5, 5],
             "score": 0.4},
            {"image_id": 2, "category_id": 3, "bbox": SBOX, "score": 0.7}]
    kept = agnostic_nms(dets, 0.65)
    assert sorted((d["image_id"], d["score"]) for d in kept) == \
        [(1, 0.4), (1, 0.9), (2, 0.7)]
    assert len(agnostic_nms(dets, 1.0)) == 4


def test_camera_class_dist_normalised_max_per_class():
    dist = camera_class_dist([
        {"image_id": 7, "category_id": 1, "bbox": CBOX, "score": 0.6},
        {"image_id": 7, "category_id": 1, "bbox": CBOX, "score": 0.3},
        {"image_id": 7, "category_id": 2, "bbox": CBOX, "score": 0.2}])
    assert dist[7] == {1: pytest.approx(0.75), 2: pytest.approx(0.25)}


# ---------------------------------------------------------------- CLI

def _write_root(tmp_path):
    paired = tmp_path / "paired"
    paired.mkdir()
    son = _gt(0, SBOX)
    cam = _gt(100, CBOX)
    # frame 7: a train frame with GT and junk predictions -> must be ignored
    son["images"].append({"id": 7, "width": 1920, "height": 1080})
    son["annotations"].append({"id": 7, "image_id": 7, "category_id": 1,
                               "bbox": SBOX, "area": 400, "iscrowd": 0,
                               "source": "blob"})
    cam["images"].append({"id": 107, "width": 1920, "height": 1080})
    json.dump(son, open(paired / "sonar.json", "w"))
    json.dump(cam, open(paired / "camera.json", "w"))
    with open(paired / "frame_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "camera_image_id", "sonar_image_id",
                    "split", "recording"])
        for s in list(GT_CLS) + [7]:
            w.writerow([f"s{s}", s + 100, s, "val" if s <= 6 else "train",
                        f"rec{(s + 1) // 2}"])
    sp = tmp_path / "pred.sonar.d0.json"
    cp = tmp_path / "pred.camera.d0.json"
    json.dump(SONAR_ALL_CLS1 + [{"image_id": 7, "category_id": 2,
                                 "bbox": [0, 0, 3, 3], "score": 0.99}],
              open(sp, "w"))
    json.dump(_cam(GT_CLS), open(cp, "w"))
    return paired, sp, cp


def _cli(*args):
    return subprocess.run([sys.executable, "tools/class_transplant.py",
                           *args], cwd=REPO, capture_output=True, text=True)


def test_cli_end_to_end(tmp_path):
    paired, sp, cp = _write_root(tmp_path)
    out = tmp_path / "t.json"
    r = _cli("--sonar-pred", str(sp), "--camera-pred", str(cp),
             "--paired", str(paired), "--split", "val", "--out", str(out))
    assert r.returncode == 0, r.stderr
    res = json.load(open(out))
    assert res["diagnostics"]["n_eval_images"] == 6
    assert res["pooled"]["camera_hard"]["AP50"] == pytest.approx(1.0)
    assert res["pooled"]["original"]["AP50"] == pytest.approx(1 / 3,
                                                              abs=1e-6)
    recs = {row["group"] for row in res["breakdown"]["camera_hard"]}
    assert recs == {"all", "rec1", "rec2", "rec3"}


def test_cli_refuses_test_split(tmp_path):
    paired, sp, cp = _write_root(tmp_path)
    r = _cli("--sonar-pred", str(sp), "--camera-pred", str(cp),
             "--paired", str(paired), "--split", "val,test",
             "--out", str(tmp_path / "t.json"))
    assert r.returncode != 0
    assert "sealed" in r.stderr
    assert not (tmp_path / "t.json").exists()
