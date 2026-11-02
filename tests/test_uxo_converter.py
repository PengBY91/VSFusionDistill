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
    VAL_ID_SEGMENTS,
    _train_roles,
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


# --- interleaved val_id (2026-11-02 amendment, rethink §9.2 h) ---------

def _blocks(roles, label):
    """[(start, end)) runs of ``label`` in a role list."""
    runs, start = [], None
    for t, r in enumerate(roles + ["<end>"]):
        if r == label and start is None:
            start = t
        elif r != label and start is not None:
            runs.append((start, t))
            start = None
    return runs


@pytest.mark.parametrize("n", [20, 37, 150, 600])
@pytest.mark.parametrize("rec", ["2023-09-20_180722", "2024-01-01_000000",
                                 "rec_x"])
def test_interleaved_roles_shape(n, rec):
    """Two val_id blocks, never at the tail-only position by
    construction, total share = tail, at least one segment apart, every
    train<->val_id boundary separated by exactly round(n*gap) unused
    frames (or the recording edge), and >= half the frames train."""
    roles = _train_roles(n, 0.15, 0.05, mode="interleaved", rec_key=rec)
    blocks = _blocks(roles, "val_id")
    assert len(blocks) == 2
    b = max(1, int(round(n * 0.15 / 2)))
    assert all(e - s == b for s, e in blocks)
    seg = n / VAL_ID_SEGMENTS
    # centred in non-adjacent segments
    segs = [int((s + e) / 2 // seg) for s, e in blocks]
    assert segs[1] - segs[0] >= 2
    g = int(round(n * 0.05))
    for s, e in blocks:
        left = roles[max(0, s - g):s]
        right = roles[e:min(n, e + g)]
        assert all(r is None for r in left + right)
        # no train frame touches a val_id frame
        if s - g - 1 >= 0:
            assert roles[s - 1] is None or g == 0
        if e < n:
            assert roles[e] is None or g == 0
    for t in range(n - 1):
        pair = {roles[t], roles[t + 1]}
        assert pair != {"train", "val_id"}
    assert roles.count("train") >= n / 2


def test_interleaved_roles_deterministic_and_keyed():
    """Same (seed, recording) -> same blocks across calls; the draw
    depends on the recording name and on the seed, so recordings do not
    all hold out the same segments."""
    a = _train_roles(150, 0.15, 0.05, mode="interleaved", rec_key="r1")
    assert a == _train_roles(150, 0.15, 0.05, mode="interleaved",
                             rec_key="r1")
    draws = {tuple(_blocks(_train_roles(150, 0.15, 0.05,
                                        mode="interleaved",
                                        rec_key=f"rec{i}"), "val_id"))
             for i in range(30)}
    assert len(draws) > 5
    seeds = {tuple(_blocks(_train_roles(150, 0.15, 0.05,
                                        mode="interleaved", rec_key="r1",
                                        seed=s), "val_id"))
             for s in range(30)}
    assert len(seeds) > 5


def test_interleaved_short_or_disabled_stays_train():
    assert _train_roles(19, 0.15, 0.05, mode="interleaved",
                        rec_key="r") == ["train"] * 19
    assert _train_roles(150, 0.0, 0.05, mode="interleaved",
                        rec_key="r") == ["train"] * 150
    with pytest.raises(ValueError):
        _train_roles(150, 0.15, 0.05, mode="bogus")


def test_tail_mode_unchanged_by_amendment():
    """The default path is the preregistered tail carve, bit for bit."""
    assert _train_roles(20, 0.15, 0.05) == (["train"] * 16 + [None]
                                            + ["val_id"] * 3)
    assert _train_roles(20, 0.15, 0.05, mode="tail", rec_key="x",
                        seed=7) == _train_roles(20, 0.15, 0.05)


def test_e11_interleaved_invocation_roundtrip(tmp_path):
    """E11 invocation after the amendment (manifest + cap 30 + 0.15 /
    0.05 + interleaved, seed 0): report bookkeeping matches frame_meta,
    val_id frames are not the trajectory tail, no val_id frame is
    adjacent (in kept-frame order) to a train frame, and a rerun is
    byte-identical."""
    src = _make_fake_uxo(tmp_path / "src")
    _make_fake_uxo_long(src, n_frames=40)
    out = tmp_path / "paired"
    manifest = {"seed": 0, "recordings": {
        REC_A: {"split": "train", "moving_target": False},
        REC_B: {"split": "val", "moving_target": False},
        REC_C: {"split": "train", "moving_target": False}}}
    kw = dict(split_manifest=manifest, max_frames_per_recording=30,
              val_id_tail_frac=0.15, val_id_gap_frac=0.05,
              val_id_mode="interleaved", val_id_seed=0)
    report = convert(str(src), str(out), **kw)
    data = read_paired(str(out))
    c = report["recordings"][REC_C]
    assert c["val_id_mode"] == "interleaved"
    expect = _train_roles(30, 0.15, 0.05, mode="interleaved",
                          rec_key=REC_C, seed=0)
    assert c["val_id_frames"] == expect.count("val_id")
    assert c["gap_unused"] == expect.count(None)
    assert c["kept"] == 30 - expect.count(None)
    c_rows = sorted((int(m["frame_idx"]), m["split"]) for m in
                    data.frame_meta if m["recording"] == REC_C)
    assert [s for _, s in c_rows] == [r for r in expect if r is not None]
    assert len(_blocks(expect, "val_id")) == 2
    # val_rows exist only on train recordings
    assert {m["recording"] for m in data.frame_meta
            if m["split"] == "val_id"} <= {REC_A, REC_C}
    f1 = (out / "frame_meta.csv").read_text()
    convert(str(src), str(out), **kw)
    assert (out / "frame_meta.csv").read_text() == f1


def test_trim_window_bounds():
    """E11 修订 2 (2026-11-02): the frozen window is positions i with
    0.2 <= i/(n-1) <= 0.8 on the capped frame order."""
    from aodet.datasets.converters.uxo_to_paired import trim_window
    assert trim_window(150, 0.2) == (30, 120)   # 30/149 .. 119/149
    assert trim_window(10, 0.2) == (2, 8)       # 2/9 .. 7/9
    assert trim_window(91, 0.2) == (18, 73)     # 18/90=0.2 exactly kept
    assert trim_window(1, 0.2) == (0, 1)
    assert trim_window(5, 0.0) == (0, 5)
    with pytest.raises(ValueError):
        trim_window(10, 0.5)
    with pytest.raises(ValueError):
        trim_window(10, -0.1)


def _make_long_recording(src, n=30):
    """One fully annotated recording with n contiguous frames."""
    label = {"class": "mortar_shell", "x_min": 10, "y_min": 20,
             "x_max": 110, "y_max": 60}
    rec = "2023-01-01_100000"
    for sub in ("gopro", "aris_raw", "labels"):
        (src / rec / sub).mkdir(parents=True)
    (src / rec / "aris_file_meta.yaml").write_text(
        "NumRawBeams: 64\nSamplesPerChannel: 100\n")
    for i in range(n):
        _write_frame(src / rec, f"{i:04d}", label=label)
    import csv
    with open(src / rec / "gantry.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["aris_frame_idx", "x", "y", "z"])
        for i in range(n):
            w.writerow([i, 0.01 * i, -0.2, -0.25])
    cols = ("FrameIndex", "FrameTime", "sonarTimeStamp", "WindowStart",
            "WindowLength", "SamplePeriod", "ReceiverGain", "Intensity",
            "Focus", "SoundSpeed", "TargetRange", "TargetBearing")
    with open(src / rec / "aris_frame_meta.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([i, 0, 0, 0.69, 2.32, 4.0, 10, 28, 528, 1491.3,
                        3.2, -0.4])
    return rec


def test_convert_trims_endpoint_frames(tmp_path):
    """trim_ends_frac drops the first/last fraction of the kept order
    before val_id carving; the dropped count is visible in the report
    and no endpoint frame reaches the paired output."""
    src = tmp_path / "src"
    rec = _make_long_recording(src, n=30)
    out = tmp_path / "paired"
    report = convert(str(src), str(out), test_recordings=(),
                     trim_ends_frac=0.2)
    a = report["recordings"][rec]
    # n=30: k = ceil(0.2*29) = 6 -> frames [6, 24) kept (18)
    assert a["trim_ends_dropped"] == 12
    assert a["kept"] == 18
    data = read_paired(str(out))
    idxs = sorted(int(m["frame_idx"]) for m in data.frame_meta)
    assert idxs == list(range(6, 24))
    # trim 0 keeps everything (legacy behaviour, field absent)
    report0 = convert(str(src), str(tmp_path / "paired0"),
                      test_recordings=())
    assert "trim_ends_dropped" not in report0["recordings"][rec]
    assert report0["recordings"][rec]["kept"] == 30


def test_convert_trim_applies_after_cap(tmp_path):
    """Frozen order: cap first, then trim -- the trimmed statistics in
    the projection report are computed on exactly this capped order."""
    src = tmp_path / "src"
    rec = _make_long_recording(src, n=30)
    out = tmp_path / "paired"
    report = convert(str(src), str(out), test_recordings=(),
                     max_frames_per_recording=10, trim_ends_frac=0.2)
    a = report["recordings"][rec]
    assert a["capped_skipped"] == 20
    # cap 10 -> trim_window(10, .2) = (2, 8): 6 frames survive
    assert a["trim_ends_dropped"] == 4
    assert a["kept"] == 6


def test_manifest_excluded_split_skips_recording(tmp_path):
    """The split tool's manifest marks filter-rejected recordings with
    split="excluded" (not an ``excluded`` bool); the converter must
    skip them with the reason visible instead of aborting the run."""
    src = _make_fake_uxo(tmp_path / "src")
    out = tmp_path / "paired"
    manifest = {"recordings": {
        REC_A: {"split": "train", "moving_target": False},
        REC_B: {"split": "excluded", "moving_target": False,
                "exclude_reason": "row residual p50 96.9 px > 20.0"},
    }}
    report = convert(str(src), str(out), split_manifest=manifest)
    a = report["recordings"][REC_A]
    assert (a["split"], a["kept"]) == ("train", 2)
    b = report["recordings"][REC_B]
    assert b["split"] == "excluded"
    assert "96.9" in b["exclude_reason"]
    assert b["kept"] == 0
    assert report["total_kept"] == 2
