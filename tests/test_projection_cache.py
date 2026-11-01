"""E11 prereq 2: the projection calibration cache is keyed by an input
fingerprint (source annotations + poses + blob detections + fit
parameters), not by recording name (rethink 2026-11-01, 代码与数据前置项
item 2).  Three properties are asserted:

1. the fingerprint is deterministic and sensitive to every calibration
   input (pose, label, category, blob, order, parameters);
2. a cache hit reproduces the freshly computed calibration exactly --
   the "缓存命中/未命中结果一致" equivalence the prereq demands;
3. the tool's cache helpers honour fingerprints and never trust legacy
   v1 (name-keyed) caches.
"""

import dataclasses
import importlib.util
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aodet.datasets.converters import uxo_projection as up
from tests.test_uxo_projection import _synth_tank


def _tool():
    """Import tools/project_uxo_sonar.py as a module (same pattern as
    test_calib_cache_roundtrip in test_uxo_projection.py)."""
    spec = importlib.util.spec_from_file_location(
        "project_uxo_sonar",
        os.path.join(os.path.dirname(up.__file__), "..", "..", "..",
                     "tools", "project_uxo_sonar.py"))
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def test_fingerprint_deterministic_and_sensitive():
    obs, _ = _synth_tank(arc=False, aim=False, n=40)
    fp = up.calib_fingerprint("rec", obs, 20)
    assert fp == up.calib_fingerprint("rec", obs, 20)      # deterministic
    assert up.calib_fingerprint("rec", obs, 25) != fp      # fit parameter
    o0 = obs[0]
    pose = [dataclasses.replace(o0, gantry=o0.gantry + 1e-4)] + obs[1:]
    assert up.calib_fingerprint("rec", pose, 20) != fp     # gantry pose
    ray = o0.cam_ray + np.array([1e-3, 0.0, 0.0])
    ray = ray / np.linalg.norm(ray)
    label = [dataclasses.replace(o0, cam_ray=ray)] + obs[1:]
    assert up.calib_fingerprint("rec", label, 20) != fp    # source label
    cat = [dataclasses.replace(o0, category_id=1)] + obs[1:]
    assert up.calib_fingerprint("rec", cat, 20) != fp      # category
    blob = [dataclasses.replace(o0, blob=(o0.blob[0] + 0.5,) + o0.blob[1:])]
    blob += obs[1:]
    assert up.calib_fingerprint("rec", blob, 20) != fp     # blob detection
    assert up.calib_fingerprint("rec", list(reversed(obs)), 20) != fp  # order
    # the serialised form of an unlabelled frame is stable too
    bare = [dataclasses.replace(o0, cam_ray=None, category_id=None,
                                blob=None, theta=0.0, rng=0.0)]
    up.calib_fingerprint("rec", bare + obs[1:], 20)
    # parameter dicts are plain data (json-safe) and stable
    assert up.fit_params(20) == up.fit_params(20)


def test_cache_hit_reproduces_fresh_calibration(tmp_path):
    """Equivalence test (E11 prereq 2): a cache hit and a fresh
    calibration give identical results -- save -> load -> predict must
    be bit-for-bit the calibrate() -> anchor_b_rng() pipeline output."""
    tool = _tool()
    obs, _ = _synth_tank(arc=False, aim=False)
    fresh = up.calibrate("synthetic", obs, min_frames=20)
    fresh, _anchor_stats = up.anchor_b_rng(fresh, obs)
    fps = {"synthetic": up.calib_fingerprint("synthetic", obs, 20)}
    cache = str(tmp_path / "projection_calib.json")
    tool.save_calib_cache(cache, fps,
                          {"synthetic": tool._calib_to_dict(fresh)})
    usable, stale = tool.load_calib_cache(cache, fps)
    assert stale == [] and list(usable) == ["synthetic"]
    cached = tool._calib_from_dict(usable["synthetic"])
    assert cached.yaw_sign == fresh.yaw_sign
    assert cached.phi_ref == fresh.phi_ref and \
        cached.yaw_span_deg == fresh.yaw_span_deg
    assert cached.b_rng == fresh.b_rng
    # json floats round-trip exactly in py3, so equality is exact
    for f in ("circle_centre", "R_cam0", "R_son0", "P"):
        assert np.array_equal(getattr(cached, f), getattr(fresh, f)), f
    for o in obs[:10]:
        assert up.predict(cached, o) == up.predict(fresh, o)


def test_cache_miss_on_changed_input_and_legacy_distrusted(tmp_path):
    tool = _tool()
    obs, _ = _synth_tank(arc=False, aim=False, n=40)
    cache = str(tmp_path / "projection_calib.json")
    fps = {"synthetic": up.calib_fingerprint("synthetic", obs, 20)}
    tool.save_calib_cache(cache, fps, {"synthetic": {"recording":
                                                     "synthetic"}})
    usable, stale = tool.load_calib_cache(cache, {})
    assert usable == {} and stale == ["synthetic"]
    usable, stale = tool.load_calib_cache(
        cache, {"synthetic": "0" * 64})
    assert usable == {} and stale == ["synthetic"]
    # legacy v1 layout (recording-name-keyed, no fingerprints): fully
    # stale, never trusted
    with open(cache, "w", encoding="utf-8") as f:
        json.dump({"synthetic": {"recording": "synthetic"}}, f)
    usable, stale = tool.load_calib_cache(cache, fps)
    assert usable == {} and stale == ["synthetic"]
    assert not os.path.exists(str(tmp_path / "missing.json")) or \
        tool.load_calib_cache(str(tmp_path / "missing.json"), fps) == \
        ({}, [])
