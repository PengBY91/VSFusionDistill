"""Offline degraded test sets (tools/make_degraded_testsets.py).

D1 camera pixels must differ from D0, D0 must be a byte-identical copy,
sonar files must be hardlinks, and the manifest must pin every file's
sha256.  Degradation itself is deterministic: running the tool twice
gives identical hashes.
"""

import csv
import json
import os
import subprocess
import sys

import pytest

cv2 = pytest.importorskip("cv2")

from tests.test_paired_pipeline import _make_toy_paired

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(root, out):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, "tools",
                                      "make_degraded_testsets.py"),
         root, out, "--splits", "test"],
        capture_output=True, text=True, cwd=REPO)
    assert r.returncode == 0, r.stdout + r.stderr


def test_offline_levels_and_manifest(tmp_path):
    root = str(tmp_path / "paired")
    _make_toy_paired(root, n=4, split="test")
    out = str(tmp_path / "offline")
    _run(root, out)

    for level in ("d0", "d1", "d2", "d3", "d4"):
        assert os.path.isdir(os.path.join(out, level))
        with open(os.path.join(out, level, "frame_meta.csv")) as f:
            rows = list(csv.DictReader(f))
        assert rows and all(r["deg_level"] == level for r in rows)
    # camera pixels: D0 identical, D2 different
    import numpy as np
    d0 = cv2.imread(os.path.join(out, "d0", "images", "camera", "c000.jpg"))
    src = cv2.imread(os.path.join(root, "images", "camera", "c000.jpg"))
    d2 = cv2.imread(os.path.join(out, "d2", "images", "camera", "c000.jpg"))
    assert np.array_equal(d0, src)
    assert not np.array_equal(d2, src)

    # sonar hardlinked
    src_s = os.path.join(root, "images", "sonar", "s000.png")
    out_s = os.path.join(out, "d2", "images", "sonar", "s000.png")
    assert os.stat(src_s).st_ino == os.stat(out_s).st_ino

    # manifest pins hashes and matches reality
    with open(os.path.join(out, "manifest.json")) as f:
        m = json.load(f)
    import hashlib
    for rel, sha in m["d2"]["files"].items():
        p = os.path.join(out, "d2", rel)
        assert os.path.exists(p), rel
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        assert h == sha, rel

    # determinism: same input -> same hashes
    out2 = str(tmp_path / "offline2")
    _run(root, out2)
    with open(os.path.join(out2, "manifest.json")) as f:
        m2 = json.load(f)
    assert m["d2"]["files"] == m2["d2"]["files"]
