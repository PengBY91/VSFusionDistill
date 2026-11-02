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
    assert a["category_dir"] == ""              # flat sample layout
    b = report["recordings"][REC_B]
    assert (b["split"], b["kept"]) == ("test", 2)
    assert b["no_gantry"] == 1                  # 0202
    assert report["total_kept"] == 4
    assert report["total_annotations"] == 4
    assert "evil..dir" not in report["recordings"]


def test_class_grouped_layout(tmp_path):
    """data_export_recordings.7z nests recordings one level deeper:
    <src>/<category_dir>/<recording>/ (94 recordings under 5 category
    dirs, verified against the archive listing 2026-11-02).  Discovery
    must walk the extra level, tag the report, and keep full
    provenance in source_path_* -- while per-recording accounting
    stays identical to the flat layout."""
    src = _make_fake_uxo(tmp_path / "grouped" / "15cm_mortar")
    out = tmp_path / "paired"
    report = convert(str(src.parent), str(out), test_recordings=(REC_B,))
    a = report["recordings"][REC_A]
    assert (a["split"], a["kept"]) == ("train", 2)
    assert a["classes"] == {"mortar_shell": 2}
    assert a["category_dir"] == "15cm_mortar"
    b = report["recordings"][REC_B]
    assert (b["split"], b["kept"]) == ("test", 2)
    assert b["category_dir"] == "15cm_mortar"
    assert report["total_kept"] == 4
    assert report["total_annotations"] == 4
    assert "evil..dir" not in report["recordings"]
    data = read_paired(str(out))
    by_sid = {m["sample_id"]: m for m in data.frame_meta}
    m = by_sid[f"{REC_A}_0100"]
    assert m["source_path_camera"] == f"15cm_mortar/{REC_A}/gopro/0100.jpg"
    assert m["source_path_sonar"] == f"15cm_mortar/{REC_A}/aris_raw/0100.pgm"


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


# --------------------------------------------------------------------------
# E11 prereq 3 (rethink 2026-11-01, 代码与数据前置项 item 3): manifest
# three-way split, per-recording cap, val-ID tail, moving_target column
# --------------------------------------------------------------------------
REC_C = "2023-01-03_110000"


def _make_fake_uxo_long(src, n_frames=20, rec=REC_C):
    """One recording with n_frames consecutive fully-annotated frames."""
    for sub in ("gopro", "aris_raw", "labels"):
        (src / rec / sub).mkdir(parents=True, exist_ok=True)
    (src / rec / "aris_file_meta.yaml").write_text(
        "NumRawBeams: 64\nSamplesPerChannel: 100\n")
    label = {"class": "mortar_shell", "x_min": 100, "y_min": 50,
             "x_max": 200, "y_max": 90}
    idxs = [f"{i:04d}" for i in range(n_frames)]
    for i in idxs:
        _write_frame(src / rec, i, label=label)
    with open(src / rec / "gantry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aris_frame_idx", "x", "y", "z"])
        for i in idxs:
            w.writerow([int(i), 0.1 * int(i), -0.2, -0.25])
    with open(src / rec / "aris_frame_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["FrameIndex", "FrameTime", "sonarTimeStamp",
                    "WindowStart", "WindowLength", "SamplePeriod",
                    "ReceiverGain", "Intensity", "Focus", "SoundSpeed",
                    "TargetRange", "TargetBearing"])
        for i in idxs:
            w.writerow([int(i), 1695233343329067, 1695233252025080,
                        0.69, 2.32, 4.0, 10, 28, 528, 1491.3, 3.2, -0.4])
    return src


def test_manifest_three_way_split_and_moving_target(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    manifest = {"recordings": {
        REC_A: {"split": "train", "moving_target": False},
        REC_B: {"split": "val", "moving_target": True}}}
    report = convert(str(src), str(out), split_manifest=manifest)
    data = read_paired(str(out))
    by_sid = {m["sample_id"]: m for m in data.frame_meta}
    assert by_sid[f"{REC_A}_0100"]["split"] == "train"
    assert by_sid[f"{REC_A}_0100"]["moving_target"] == "false"
    assert by_sid[f"{REC_B}_0200"]["split"] == "val"    # not test
    assert by_sid[f"{REC_B}_0200"]["moving_target"] == "true"
    assert report["recordings"][REC_A]["split_source"] == "manifest"
    assert report["recordings"][REC_B]["split"] == "val"
    # legacy default path: no manifest -> moving_target unknown ("")
    src2 = _make_fake_uxo(tmp_path / "src2")
    convert(str(src2), str(tmp_path / "paired2"), test_recordings=(REC_B,))
    data2 = read_paired(str(tmp_path / "paired2"))
    assert {m["moving_target"] for m in data2.frame_meta} == {""}


def test_train_tail_carves_val_id_with_gap(tmp_path):
    src = _make_fake_uxo_long(tmp_path / "src", n_frames=20)
    out = tmp_path / "paired"
    report = convert(str(src), str(out),
                     split_manifest={"recordings": {REC_C: {"split":
                                                           "train"}}},
                     val_id_tail_frac=0.15, val_id_gap_frac=0.05)
    data = read_paired(str(out))
    rows = sorted(data.frame_meta, key=lambda m: int(m["frame_idx"]))
    # 20 kept frames: n_tail = round(20*0.15) = 3, n_gap = round(20*0.05)
    # = 1 -> 16 train, frame 0016 dropped as the gap, 0017-0019 val_id
    assert [m["split"] for m in rows] == ["train"] * 16 + ["val_id"] * 3
    assert [int(m["frame_idx"]) for m in rows][-3:] == [17, 18, 19]
    assert not any(int(m["frame_idx"]) == 16 for m in rows)
    a = report["recordings"][REC_C]
    assert (a["kept"], a["val_id_frames"], a["gap_unused"]) == (19, 3, 1)


def test_short_train_recording_not_carved(tmp_path):
    """Fewer than 8 kept frames: no carve-out, all frames stay train
    (a 2-frame val_id block measures nothing)."""
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    report = convert(str(src), str(out),
                     split_manifest={"recordings": {REC_A: {"split":
                                                           "train"}}},
                     val_id_tail_frac=0.15, val_id_gap_frac=0.05)
    data = read_paired(str(out))
    assert {m["split"] for m in data.frame_meta
            if m["recording"] == REC_A} == {"train"}
    a = report["recordings"][REC_A]
    assert a["val_id_frames"] == 0 and a["gap_unused"] == 0


def test_cap_subsamples_evenly(tmp_path):
    src = _make_fake_uxo_long(tmp_path / "src", n_frames=20)
    out = tmp_path / "paired"
    report = convert(str(src), str(out),
                     split_manifest={"recordings": {REC_C: {"split":
                                                           "test"}}},
                     max_frames_per_recording=10)
    data = read_paired(str(out))
    kept = sorted(int(m["frame_idx"]) for m in data.frame_meta)
    # evenly spaced over the 0..19 trajectory, both ends included
    # (step 19/9: positions round(i*2.111))
    assert kept == [0, 2, 4, 6, 8, 11, 13, 15, 17, 19]
    a = report["recordings"][REC_C]
    assert (a["kept"], a["capped_skipped"]) == (10, 10)


def test_manifest_excluded_recording_skipped(tmp_path):
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    manifest = {"recordings": {
        REC_A: {"split": "train"},
        REC_B: {"split": "test", "excluded": True,
                "exclude_reason": "row residual p50 > 10 px"}}}
    report = convert(str(src), str(out), split_manifest=manifest)
    data = read_paired(str(out))
    b = report["recordings"][REC_B]
    assert b["split"] == "excluded" and b["kept"] == 0
    assert b["exclude_reason"] == "row residual p50 > 10 px"
    assert not any(m["recording"] == REC_B for m in data.frame_meta)
    assert any(m["recording"] == REC_A for m in data.frame_meta)


def test_e11_invocation_roundtrip_and_deterministic(tmp_path):
    """往返测试 (E11 prereq 3): the full E11 invocation (manifest +
    cap 30 + tail 0.15 / gap 0.05) round-trips through read_paired with
    the bookkeeping the report claims, and a rerun is byte-identical."""
    src = _make_fake_uxo(tmp_path / "src")
    _make_fake_uxo_long(src, n_frames=40)
    out = tmp_path / "paired"
    manifest = {"seed": 0, "recordings": {
        REC_A: {"split": "train", "moving_target": False},
        REC_B: {"split": "val", "moving_target": True},
        REC_C: {"split": "train", "moving_target": True}}}
    report = convert(str(src), str(out), split_manifest=manifest,
                     max_frames_per_recording=30,
                     val_id_tail_frac=0.15, val_id_gap_frac=0.05)
    data = read_paired(str(out))
    assert {m["split"] for m in data.frame_meta} <= {"train", "val",
                                                     "val_id"}
    # val_id rows live on train recordings only
    assert {m["recording"] for m in data.frame_meta
            if m["split"] == "val_id"} <= {REC_A, REC_C}
    for rec, meta in manifest["recordings"].items():
        rows = [m for m in data.frame_meta if m["recording"] == rec]
        assert rows, rec
        assert {m["moving_target"] for m in rows} == (
            {"true"} if meta["moving_target"] else {"false"})
        assert len(rows) == report["recordings"][rec]["kept"]
    # REC_C: 40 candidates -> cap 30 -> roles on 30 kept (banker's
    # rounding): n_tail = round(4.5) = 4, n_gap = round(1.5) = 2
    c = report["recordings"][REC_C]
    assert (c["kept"], c["val_id_frames"], c["gap_unused"]) == (28, 4, 2)
    c_rows = sorted((int(m["frame_idx"]), m["split"]) for m in
                    data.frame_meta if m["recording"] == REC_C)
    assert [s for _, s in c_rows].count("val_id") == 4
    # the val_id block is the trajectory tail (largest frame indices)
    assert all(s == "val_id" for _, s in c_rows[-4:])
    # rerun determinism
    c1 = (out / "camera.json").read_text()
    f1 = (out / "frame_meta.csv").read_text()
    convert(str(src), str(out), split_manifest=manifest,
            max_frames_per_recording=30, val_id_tail_frac=0.15,
            val_id_gap_frac=0.05)
    assert (out / "camera.json").read_text() == c1
    assert (out / "frame_meta.csv").read_text() == f1
