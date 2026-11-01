"""Known-answer tests for the split/breakdown CLI of the anatomy tools
(E11 prereg item 5, doc/experiments.md §E11).

tools/anatomy.py and tools/camera_anatomy.py run as subprocesses on a
tiny synthetic paired root (two recordings across two splits, two
categories, one annotation per frame — the E11 split scheme at toy
scale): perfect predictions must give AP50 = 1.0 in every per-recording
and per-class cell, fully swapped classes must give class-aware 0.0 with
the class-agnostic column unchanged.  tools/test.py cannot be run
headless here (it needs a checkpoint), so its split filter and recording
map are exercised as pure functions instead.
"""

import csv
import importlib.util
import json
import os
import subprocess
import sys

import pytest

pycocotools = pytest.importorskip("pycocotools")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CATEGORIES = [{"id": 1, "name": "mortar"}, {"id": 2, "name": "grenade"}]

# (image_id, category_id, bbox); sonar and camera share the layout with
# their own id spaces (the paired format keeps them separate)
_FRAMES = [
    # recording recT / split test
    (1, 1, [100, 100, 80, 60]),
    (2, 2, [200, 120, 80, 60]),
    # recording recV / split val
    (3, 1, [120, 140, 80, 60]),
    (4, 2, [220, 160, 80, 60]),
]
_SPLIT = {1: "test", 2: "test", 3: "val", 4: "val"}
_REC = {1: "20230101_recT", 2: "20230101_recT",
        3: "20230102_recV", 4: "20230102_recV"}


def _perfect_preds(image_ids):
    return [{"image_id": i, "category_id": c, "bbox": b, "score": 0.9}
            for i, c, b in _FRAMES if i in image_ids]


def _swapped_preds(image_ids):
    return [dict(d, category_id=3 - d["category_id"])
            for d in _perfect_preds(image_ids)]


def _write_coco(path, name):
    with open(path, "w") as f:
        json.dump({
            "images": [{"id": i, "width": 640, "height": 480,
                        "file_name": f"{name}{i}.png"} for i in _SPLIT],
            "annotations": [
                {"id": n, "image_id": i, "category_id": c, "bbox": b,
                 "area": b[2] * b[3], "iscrowd": 0}
                for n, (i, c, b) in enumerate(_FRAMES, start=1)],
            "categories": CATEGORIES}, f)


def _write_frame_meta(root):
    with open(os.path.join(root, "frame_meta.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "sample_id", "camera_image_id", "sonar_image_id", "split",
            "recording"])
        w.writeheader()
        for i in _SPLIT:
            w.writerow(dict(sample_id=f"s{i:03d}", camera_image_id=i,
                            sonar_image_id=i, split=_SPLIT[i],
                            recording=_REC[i]))


def make_paired_root(root):
    """Synthetic paired root + perfect/swapped run dirs with pred dumps."""
    _write_coco(os.path.join(root, "sonar.json"), "s")
    _write_coco(os.path.join(root, "camera.json"), "c")
    _write_frame_meta(root)
    for preds, tag in ((_perfect_preds(_SPLIT), "perfect"),
                       (_swapped_preds(_SPLIT), "swapped")):
        run = os.path.join(root, tag)
        os.makedirs(run, exist_ok=True)
        for branch in ("sonar_uncoupled", "camera"):
            with open(os.path.join(run, f"pred.{branch}.d0.json"), "w") as f:
                json.dump(preds, f)
    return root


def _run_tool(args):
    r = subprocess.run([sys.executable] + args, capture_output=True,
                       text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# -- tools/anatomy.py ------------------------------------------------------

def test_anatomy_split_filter_and_perfect_known_answer(tmp_path):
    root = make_paired_root(str(tmp_path))
    out = str(tmp_path / "a.csv")
    _run_tool(["tools/anatomy.py", f"{root}/perfect", "--paired", root,
               "--split", "test", "--out", out])
    rows = _read_csv(out)
    assert [r["recording"] for r in rows] == ["20230101_recT"]
    assert rows[0]["AP50"] == "1.0"
    assert rows[0]["AP50_class_agnostic"] == "1.0"
    by_class = _read_csv(str(tmp_path / "a_by_class.csv"))
    assert {(r["recording"], r["category"]) for r in by_class} == {
        ("20230101_recT", "mortar"), ("20230101_recT", "grenade"),
        ("20230101_recT", "all"), ("all", "mortar"), ("all", "grenade"),
        ("all", "all")}
    assert all(float(r["AP50"]) == 1.0 for r in by_class)


def test_anatomy_comma_split_unions_splits(tmp_path):
    root = make_paired_root(str(tmp_path))
    out = str(tmp_path / "a.csv")
    _run_tool(["tools/anatomy.py", f"{root}/perfect", "--paired", root,
               "--split", "test,val", "--out", out])
    rows = _read_csv(out)
    assert [r["recording"] for r in rows] == ["20230101_recT",
                                              "20230102_recV"]
    # split all must give the same union when every split is selected
    out_all = str(tmp_path / "a_all.csv")
    _run_tool(["tools/anatomy.py", f"{root}/perfect", "--paired", root,
               "--split", "all", "--out", out_all])
    assert _read_csv(out_all) == rows


def test_anatomy_swapped_classes_known_answer(tmp_path):
    root = make_paired_root(str(tmp_path))
    out = str(tmp_path / "a.csv")
    _run_tool(["tools/anatomy.py", f"{root}/swapped", "--paired", root,
               "--split", "test,val", "--out", out])
    rows = _read_csv(out)
    assert all(r["AP50"] == "0.0" for r in rows)               # class-aware
    assert all(r["AP50_class_agnostic"] == "1.0" for r in rows)  # unchanged
    by_class = _read_csv(str(tmp_path / "a_by_class.csv"))
    for r in by_class:
        if r["category"] != "all":
            assert float(r["AP50"]) == 0.0
        else:
            assert float(r["AP50"]) == 0.0
            assert float(r["AP50_class_agnostic"]) == 1.0


# -- tools/camera_anatomy.py ----------------------------------------------

def test_camera_anatomy_split_and_perfect_known_answer(tmp_path):
    root = make_paired_root(str(tmp_path))
    run = os.path.join(root, "perfect")
    stdout = _run_tool(["tools/camera_anatomy.py", run, "--paired", root,
                        "--level", "d0"])
    assert "@ d0 (test)" in stdout                 # default split = test
    art = json.load(open(f"{run}/camera_anatomy.d0.json"))
    rec_rows = {(r["recording"], r["category"]): r for r in art
                if r["recording"] != "all"}
    assert set(rec_rows) == {("recT", "cat1"), ("recT", "cat2"),
                             ("recT", "agnostic")}
    assert all(r["AP50"] == pytest.approx(1.0) for r in art)


def test_camera_anatomy_split_val_and_comma(tmp_path):
    root = make_paired_root(str(tmp_path))
    run = os.path.join(root, "perfect")
    stdout = _run_tool(["tools/camera_anatomy.py", run, "--paired", root,
                        "--split", "val"])
    assert "@ d0 (val)" in stdout
    assert "[recT]" not in stdout and "[recV]" in stdout
    art = json.load(open(f"{run}/camera_anatomy.d0.json"))
    assert {r["recording"] for r in art} == {"recV", "all"}

    stdout = _run_tool(["tools/camera_anatomy.py", run, "--paired", root,
                        "--split", "test,val"])
    assert "[recT]" in stdout and "[recV]" in stdout
    art = json.load(open(f"{run}/camera_anatomy.d0.json"))
    assert {r["recording"] for r in art} == {"recT", "recV", "all"}


def test_camera_anatomy_swapped_classes_known_answer(tmp_path):
    root = make_paired_root(str(tmp_path))
    run = os.path.join(root, "swapped")
    _run_tool(["tools/camera_anatomy.py", run, "--paired", root,
               "--split", "test,val"])
    art = {(r["recording"], r["category"]): r["AP50"]
           for r in json.load(open(f"{run}/camera_anatomy.d0.json"))}
    for (rec, cat), v in art.items():
        if cat == "agnostic":
            assert v == pytest.approx(1.0), (rec, cat)   # boxes all right
        else:
            assert v == 0.0, (rec, cat)                  # classes all wrong


# -- tools/test.py pure helpers -------------------------------------------

def _load_tools_test():
    pytest.importorskip("mmengine")
    spec = importlib.util.spec_from_file_location(
        "tools_test", os.path.join(REPO, "tools", "test.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_filter_infos_by_split(tmp_path):
    mod = _load_tools_test()
    infos = [
        {"sample_id": "a", "meta": {"split": "val", "recording": "r1"}},
        {"sample_id": "b", "meta": {"split": "val_id", "recording": "r1"}},
        {"sample_id": "c", "meta": {"split": "train", "recording": "r2"}},
        {"sample_id": "d", "meta": {}},
        {"sample_id": "e"},
    ]
    kept = mod.filter_infos_by_split(infos, ("val", "val_id"))
    assert [d["sample_id"] for d in kept] == ["a", "b"]
    kept = mod.filter_infos_by_split(infos, ["train"])
    assert [d["sample_id"] for d in kept] == ["c"]
    # comma strings arrive pre-split by the CLI; opaque values pass through
    kept = mod.filter_infos_by_split(infos, {"val_id"})
    assert [d["sample_id"] for d in kept] == ["b"]


def test_recording_by_image_id(tmp_path):
    mod = _load_tools_test()
    infos = [
        {"coco_sonar_image_id": 7, "meta": {"recording": "r1"}},
        {"coco_sonar_image_id": 8, "meta": {}},           # no recording
        {"coco_sonar_image_id": None, "meta": {"recording": "r2"}},
        {"coco_camera_image_id": 3, "meta": {"recording": "r3"}},
    ]
    assert mod.recording_by_image_id(infos, "sonar") == {7: "r1"}
    assert mod.recording_by_image_id(infos, "camera") == {3: "r3"}
