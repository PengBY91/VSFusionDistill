"""Tests for the UXO -> paired converter on a synthetic source tree.

The synthetic tree mirrors the real sample layout exactly (verified
2026-10-30): shared frame indices, labels in 640x360 space, gantry.csv
keyed by aris_frame_idx, aris_frame_meta.csv keyed by FrameIndex,
aris_file_meta.yaml sensor constants.  The "multiply by 3" rule is
asserted explicitly (setup doc §4.5, converter row).
"""

import csv
import json
import os

import cv2
import numpy as np
import pytest

from aodet.datasets.converters.uxo_to_paired import (
    CATEGORIES,
    convert,
)
from aodet.datasets.paired_format import read_paired

REC_A = "2023-01-01_100000"   # train
REC_B = "2023-01-02_090000"   # test


def _write_frame(rec_dir, idx, *, jpg=True, pgm=True, label=None):
    if jpg:
        cv2.imwrite(str(rec_dir / "gopro" / f"{idx}.jpg"),
                    np.zeros((16, 16, 3), np.uint8))
    if pgm:
        cv2.imwrite(str(rec_dir / "aris_raw" / f"{idx}.pgm"),
                    np.zeros((10, 8), np.uint8))
    if label is not None:
        (rec_dir / "labels" / f"{idx}.json").write_text(
            json.dumps(label))


def _make_fake_uxo(src):
    for rec in (REC_A, REC_B):
        for sub in ("gopro", "aris_raw", "labels"):
            (src / rec / sub).mkdir(parents=True)
        (src / rec / "aris_file_meta.yaml").write_text(
            "NumRawBeams: 64\nSamplesPerChannel: 100\n")
        (src / rec / "notes.txt").write_text("- target: test\n")

    mortar = {"class": "mortar_shell", "x_min": 100, "y_min": 50,
              "x_max": 200, "y_max": 90}
    bomb = {"class": "100lbs_aircraft_bomb", "x_min": 10, "y_min": 20,
            "x_max": 110, "y_max": 140}

    # rec A: 0100 kept, 0101 unlabeled (skipped by default), 0102 kept,
    # 0103 label without gopro, 0104 annotated but no aris frame
    _write_frame(src / REC_A, "0100", label=mortar)
    _write_frame(src / REC_A, "0101")
    _write_frame(src / REC_A, "0102", label=mortar)
    _write_frame(src / REC_A, "0103", jpg=False, pgm=False, label=mortar)
    _write_frame(src / REC_A, "0104", pgm=False, label=mortar)
    # rec B: 0200/0201 kept, 0202 annotated but missing from gantry.csv
    _write_frame(src / REC_B, "0200", label=bomb)
    _write_frame(src / REC_B, "0201", label=bomb)
    _write_frame(src / REC_B, "0202", label=bomb)

    gantry = {"aris_frame_idx": [], "x": [], "y": [], "z": []}
    meta = {"FrameIndex": [], "FrameTime": [], "sonarTimeStamp": [],
            "WindowStart": [], "WindowLength": [], "SamplePeriod": [],
            "ReceiverGain": [], "Intensity": [], "Focus": [],
            "SoundSpeed": [], "TargetRange": [], "TargetBearing": []}
    rows = {REC_A: ["0100", "0101", "0102", "0104"],  # 0103 has no files
            REC_B: ["0200", "0201"]}                   # 0202: no gantry row
    for rec, idxs in rows.items():
        with open(src / rec / "gantry.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["aris_frame_idx", "x", "y", "z"])
            for i in idxs:
                w.writerow([int(i), 0.1 * int(i), -0.2, -0.25])
        with open(src / rec / "aris_frame_meta.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(list(meta))
            for i in idxs:
                w.writerow([int(i), 1695233343329067, 1695233252025080,
                            0.69, 2.32, 4.0, 10, 28, 528, 1491.3, 3.2,
                            -0.4])
    # decoys the path whitelist must reject
    (src / "evil..dir").mkdir()
    (src / "evil..dir" / "labels").mkdir()
    (src / "evil..dir" / "labels" / "0300.json").write_text("{}")
    (src / REC_A / "labels" / "zzzz.json").write_text("{}")
    return src


def test_convert_counts_and_skips(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    report = convert(str(src), str(out), test_recordings=(REC_B,))
    a = report["recordings"][REC_A]
    assert (a["split"], a["kept"]) == ("train", 2)
    assert a["unlabeled_skipped"] == 1          # 0101
    assert a["label_without_gopro"] == 1        # 0103
    assert a["annotated_without_aris"] == 1     # 0104
    assert a["classes"] == {"mortar_shell": 2}
    b = report["recordings"][REC_B]
    assert (b["split"], b["kept"]) == ("test", 2)
    assert b["no_gantry"] == 1                  # 0202
    assert report["total_kept"] == 4
    assert report["total_annotations"] == 4
    assert "evil..dir" not in report["recordings"]


def test_convert_x3_boxes_and_categories(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    convert(str(src), str(out), test_recordings=(REC_B,))
    data = read_paired(str(out))
    by_img = {a["image_id"]: a for a in data.camera["annotations"]}
    # sorted category order: bomb=1, grenade=2, mortar=3
    cat_id = {c["name"]: c["id"] for c in data.camera["categories"]}
    assert cat_id == {n: i for i, n in enumerate(CATEGORIES, 1)}
    imgs = {im["file_name"]: im for im in data.camera["images"]}
    # rec A frame 0100: 100,50,200,90 (640x360 space) -> x3 in 1920x1080
    key = "images/camera/2023-01-01_100000_0100.jpg"
    ann = by_img[imgs[key]["id"]]
    assert ann["bbox"] == [300, 150, 300, 120]
    assert ann["area"] == 300 * 120
    assert ann["category_id"] == cat_id["mortar_shell"]
    # rec B frame 0200 is a bomb box: 10,20,110,140 -> 30,60,300,360
    key = "images/camera/2023-01-02_090000_0200.jpg"
    ann = by_img[imgs[key]["id"]]
    assert ann["bbox"] == [30, 60, 300, 360]
    # sonar side: no annotations yet (projection tool fills them), but
    # categories exist and image geometry follows aris_file_meta.yaml
    assert data.sonar["annotations"] == []
    assert len(data.sonar["categories"]) == 3
    s = data.samples()[0]
    son = [im for im in data.sonar["images"] if im["id"] == s["sonar_image_id"]][0]
    assert (son["width"], son["height"]) == (64, 100)
    cam = [im for im in data.camera["images"] if im["id"] == s["camera_image_id"]][0]
    assert (cam["width"], cam["height"]) == (1920, 1080)
    # pairs.csv is empty until projection
    assert data.pairs == []


def test_frame_meta_carries_projection_inputs(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    convert(str(src), str(out), test_recordings=(REC_B,))
    data = read_paired(str(out))
    by_sid = {m["sample_id"]: m for m in data.frame_meta}
    m = by_sid[f"{REC_A}_0100"]
    assert m["split"] == "train"
    assert m["recording"] == REC_A
    assert int(m["frame_idx"]) == 100  # csv free-form columns read back as str
    assert float(m["gantry_x"]) == pytest.approx(10.0)
    assert float(m["aris_windowstart"]) == pytest.approx(0.69)
    assert float(m["aris_windowlength"]) == pytest.approx(2.32)
    assert float(m["aris_soundspeed"]) == pytest.approx(1491.3)
    assert m["source_path_camera"] == f"{REC_A}/gopro/0100.jpg"
    assert m["source_path_sonar"] == f"{REC_A}/aris_raw/0100.pgm"
    assert by_sid[f"{REC_B}_0200"]["split"] == "test"
    # decoy label zzzz.json must not appear
    assert not any(m["frame_idx"] == 3000 for m in data.frame_meta)


def test_images_hardlinked_or_copied(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    convert(str(src), str(out), test_recordings=(REC_B,))
    cam = out / "images" / "camera" / "2023-01-01_100000_0100.jpg"
    son = out / "images" / "sonar" / "2023-01-01_100000_0100.pgm"
    assert cam.exists() and son.exists()
    assert cam.read_bytes() == (src / REC_A / "gopro" / "0100.jpg").read_bytes()
    # same inode when hardlinking worked (same tmpfs)
    assert (cam.stat().st_ino == (src / REC_A / "gopro" / "0100.jpg"
                                   ).stat().st_ino
            or cam.stat().st_ino != 0)


def test_include_unlabeled_emits_negatives(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    report = convert(str(src), str(out), test_recordings=(REC_B,),
                     include_unlabeled=True)
    a = report["recordings"][REC_A]
    assert a["kept"] == 3            # 0100, 0101 (negative), 0102
    assert a["unlabeled_skipped"] == 0
    data = read_paired(str(out))
    by_sid = {m["sample_id"]: m for m in data.frame_meta}
    assert f"{REC_A}_0101" in by_sid
    imgs = {im["id"]: im for im in data.camera["images"]}
    ann_img_ids = {a["image_id"] for a in data.camera["annotations"]}
    neg_id = by_sid[f"{REC_A}_0101"]["camera_image_id"]
    assert neg_id not in ann_img_ids  # no fabricated GT box


def test_stride_subsamples_by_frame_index(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    report = convert(str(src), str(out), test_recordings=(REC_B,),
                     stride=2)
    data = read_paired(str(out))
    kept = sorted(int(m["frame_idx"]) for m in data.frame_meta
                  if m["recording"] == REC_A)
    assert kept == [100, 102]        # 0101 unlabeled, 0100/0102 even
    a = report["recordings"][REC_A]
    # annotated 0103/0104 never reach the stride check (no gopro/aris)
    assert a["stride_skipped"] == 0


def test_contact_sheet_written(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    sheet = tmp_path / "sheet.jpg"
    convert(str(src), str(out), test_recordings=(REC_B,),
            contact_sheet=str(sheet))
    assert sheet.exists()
    img = cv2.imread(str(sheet))
    assert img is not None and img.ndim == 3


def test_rerun_is_deterministic(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    r1 = convert(str(src), str(out), test_recordings=(REC_B,))
    c1 = (out / "camera.json").read_text()
    r2 = convert(str(src), str(out), test_recordings=(REC_B,))
    c2 = (out / "camera.json").read_text()
    assert c1 == c2 and r1 == r2
