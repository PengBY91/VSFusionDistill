"""Tests for tools/split_uxo_recordings.py (E11 frozen 划分规则):
calibration filter before split, category x geometry stratification,
6:2:2 with per-category minimums, dev recordings never in test, seed
determinism, infeasibility abort.
"""

import importlib.util
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _tool():
    spec = importlib.util.spec_from_file_location(
        "split_uxo_recordings",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "tools", "split_uxo_recordings.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _inventory(n_per_cell=8, cats=("mortar_shell", "incindiary_grenade"),
               geoms=("arc", "line"), bad_blob=(), bad_row=(),
               missing_calib=(), dev=("2023-09-21_100927",)):
    """Synthetic inventory: every recording passes calibration except
    the named bad/missing ones.  The dev recording lands in the first
    category's first geometry cell."""
    recs = {}
    k = 0
    for cat in cats:
        for geom in geoms:
            for i in range(n_per_cell):
                k += 1
                name = f"2024-01-{k:02d}_{100000 + k:06d}" if i else None
                if i == 0 and cat == cats[0] and geom == geoms[0] \
                        and dev:
                    name = dev[0]
                elif i == 0 and dev[1:]:
                    name = dev[1]
                if name is None:
                    name = f"2024-01-{k:02d}_{100000 + k:06d}"
                e = {"category": cat, "geometry": geom,
                     "moving_target": (i == 1 and geom == "line"),
                     "n_blob_frames": 100, "proj_err_row_px_p50": 2.0,
                     "rms_az_deg": 1.0}
                if name in bad_blob:
                    e["n_blob_frames"] = 12
                if name in bad_row:
                    e["proj_err_row_px_p50"] = 25.0
                if name in missing_calib:
                    e["proj_err_row_px_p50"] = None
                recs[name] = e
    return {"src": "synthetic", "recordings": recs}


def test_assign_fulfils_frozen_constraints():
    tool = _tool()
    inv = _inventory()
    manifest, infeasible = tool.assign(inv, seed=0)
    assert infeasible == []
    by_cat_split = {}
    for name, m in manifest.items():
        assert m["split"] in ("train", "val", "test", "excluded")
        by_cat_split.setdefault(m["category"], {}).setdefault(
            m["split"], []).append(name)
    for cat, splits in by_cat_split.items():
        assert len(splits.get("val", [])) >= 3
        assert len(splits.get("test", [])) >= 3
        assert len(splits.get("train", [])) >= 6
    total: dict = {}
    for splits in by_cat_split.values():
        for s, v in splits.items():
            total[s] = total.get(s, 0) + len(v)
    # ~6:2:2 over 32 recordings (8/cell, 2 cats x 2 geoms)
    assert abs(total["test"] - 0.2 * 32) <= 2
    assert abs(total["val"] - 0.2 * 32) <= 2
    # geometry stratification: both geometries present in test
    test_geoms = {manifest[n]["geometry"] for splits in
                  by_cat_split.values() for n in splits.get("test", [])}
    assert test_geoms == {"arc", "line"}


def test_dev_recordings_never_in_test():
    tool = _tool()
    dev = ("2023-09-20_180722", "2023-09-21_100927")
    inv = _inventory(dev=dev)
    manifest, infeasible = tool.assign(inv, seed=0)
    assert infeasible == []
    for name in dev:
        assert manifest[name]["split"] in ("train", "val")


def test_calibration_filter_excludes_before_split():
    tool = _tool()
    # pick names that are not the dev-recording slots (i == 0)
    plain = sorted(set(_inventory()["recordings"])
                   - set(_tool().DEV_RECORDINGS))
    bad = (plain[0], plain[1])
    inv = _inventory(bad_blob=(bad[0],), bad_row=(bad[1],))
    manifest, infeasible = tool.assign(inv, seed=0)
    assert infeasible == []
    for name in bad:
        assert manifest[name]["split"] == "excluded"
    assert "blob frames 12 < 30" in manifest[bad[0]]["exclude_reason"]
    assert "row residual" in manifest[bad[1]]["exclude_reason"]


def test_missing_calibration_aborts_as_excluded_with_reason():
    """The filter is defined on the calibration numbers; a recording
    without them must not slip into a split silently."""
    tool = _tool()
    plain = sorted(set(_inventory()["recordings"])
                   - set(_tool().DEV_RECORDINGS))
    name = plain[2]
    inv = _inventory(missing_calib=(name,))
    manifest, _ = tool.assign(inv, seed=0)
    assert manifest[name]["split"] == "excluded"
    assert "missing calibration" in manifest[name]["exclude_reason"]


def test_assign_is_seed_deterministic():
    tool = _tool()
    inv = _inventory()
    m1, _ = tool.assign(inv, seed=0)
    m2, _ = tool.assign(inv, seed=0)
    m3, _ = tool.assign(inv, seed=1)
    assert m1 == m2
    assert any(m1[n]["split"] != m3[n]["split"] for n in m1)


def test_infeasible_category_aborts_without_manifest(tmp_path):
    tool = _tool()
    # 5 recordings of one category: cannot give 3 test + 3 val + 6 train
    inv = _inventory(n_per_cell=3, cats=("mortar_shell",),
                     geoms=("arc", "line"), dev=())
    manifest, infeasible = tool.assign(inv, seed=0)
    assert manifest and infeasible
    assert any("mortar_shell" in line for line in infeasible)
    # CLI exit code 2 and no manifest written
    inv_path = tmp_path / "inv.json"
    inv_path.write_text(json.dumps(inv))
    out = tmp_path / "manifest.json"
    r = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "tools", "split_uxo_recordings.py"),
         "--inventory", str(inv_path), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 2
    assert not out.exists()
    assert "3-fold" in r.stderr


def test_cli_writes_manifest_with_sha(tmp_path):
    inv_path = tmp_path / "inv.json"
    inv_path.write_text(json.dumps(_inventory()))
    out = tmp_path / "manifest.json"
    r = subprocess.run(
        [sys.executable,
         os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "tools", "split_uxo_recordings.py"),
         "--inventory", str(inv_path), "--out", str(out)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    manifest = json.loads(out.read_text())
    assert manifest["seed"] == 0
    assert sum(manifest["counts"].values()) == 32
    assert "sha256" in r.stdout
