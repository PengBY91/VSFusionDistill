# ao-det — degradation-robust acoustic–optic detection (F1/F2)

Codebase for the F2 topic plan: degradation-robust sonar–camera fusion
detection with privileged distillation and branch independence
(`doc/topic-f2-degradation-robust-fusion-detection-distillation-2026-10.md`),
built per `doc/topic-f2-codebase-selection-and-setup-2026-10.md`.

Base stack: MMDetection 3.3.0 (YOLOX family) + mmengine 0.10.7 +
torch 2.1.2, pinned in `env/` (CPU debug variant installed here; cu121
variant for the training machine).  No third-party repo is modified:
`third_party/` is cloned read-only with pinned SHAs
(`tools/clone_third_party.sh`, `third_party/PINNED.md`).

## Layout

| path | contents |
|---|---|
| `aodet/datasets/` | unified paired format (`paired_format.py`), `PairedSonarCameraDataset`, converters, transforms (paired geometry, camera degradation D0–D∞, fork & pack) |
| `aodet/models/` | `TwoStreamYOLOX` (three heads), fusion modules, KD losses (`branch_kd`, `cross_plane_mil`), frozen teacher, `PairedDataPreprocessor` |
| `aodet/evaluation/` | per-branch COCO AP, branch error correlation (φ, ΔP_miss, Spearman, naive-Bayes fusion, ECE) |
| `aodet/geometry/` | M₀ camera→bearing mapping + flip mirroring |
| `configs/` | `_base_` (model/dataset/schedule/runtime) + `f2/` S0-lite configs |
| `tools/` | `train.py`, `test.py` (per-branch D0–D∞ scan), `make_degraded_testsets.py`, `run_seeds.sh`, `collect_results.py`, third-party pinning, RSFusionDet watch |
| `tests/` | 63 tests: degradation physics, format round-trip, geometry, metrics, model, UXO converter, end-to-end pipeline & tools |
| `env/` | install scripts + `ENVIRONMENT.md` version record |

## Quick start

```bash
source .venv/bin/activate
python -m pytest tests/ -q          # 63 passed

# S0-lite (UXO sample already converted to data/paired/uxo/)
python tools/train.py configs/f2/s0_b0_sonar_only_yoloxs.py --seed 0
python tools/test.py  configs/f2/s0_b0_sonar_only_yoloxs.py \
    work_dirs/s0_b0_sonar_only_yoloxs_seed0/latest.pth \
    --out-dir work_dirs/s0_b0_sonar_only_yoloxs_seed0
python tools/collect_results.py work_dirs --out results/results.csv
```

## Paired format (one layout for every source dataset)

```
<root>/camera.json  sonar.json  pairs.csv  frame_meta.csv
<root>/images/camera/...  images/sonar/...
```

`frame_meta.csv` keys each `sample_id` to both COCO image ids, the split
and the degradation seed; `read_paired` normalises CSV types.  Sync flips
are only legal while the sonar bearing axis is 0°-centred
(`sonar_zero_centred=True`).

## Degradation protocol

`aodet/datasets/transforms/camera_degradation.py`: physically ordered
operator (attenuation → wavelength absorption → scattering veil →
blur) with levels D0–D4, D∞ (black / sensor-absent), deterministic per
(sample_id, level, seed).  Training draws levels from p(s) (F2 §5.2)
with D∞ materialised as a black frame (`dinf_mode="zero"`, dense
batches); test-time D∞ drops the camera (`dinf_mode="drop"`,
batch 1).  Offline pinned test sets: `tools/make_degraded_testsets.py`
(sha256 manifest per level).

## Evaluation

Per-branch AP by degradation level from saved prediction files
(`tools/test.py` → `pred.<branch>.<level>.json` + `ap_summary.json`);
branch error correlation (φ, conditional miss lift ΔP_miss with
bootstrap CI, Spearman confidence correlation, naive-Bayes fused
existence ECE) in `aodet/evaluation/branch_correlation.py`.

## Status (2026-10-31)

- W1 environment installed + verified; COCO YOLOX-S reproduction gate
  passed on the full val2017 (5000 images, AP 40.30 vs official 40.5);
  W2 paired format /
  degradation / transforms / metrics done and tested; W3 model +
  configs + tools done, S0-lite smoke passes end-to-end on synthetic
  data.
- UXO sample verified (×3 label rule, GoPro alignment, extrinsics) and
  converted: 2573 pairs under `data/paired/uxo/` (recording-level
  split, train 2199 / test 374).  Sonar-side boxes + cross-modal
  correspondence π done (TODO 14.2 item 4): per-recording robust
  calibration + projection, `sonar.json` annotated (blob-gated 1185 /
  geometric 1388), `pairs.csv` written; error stats + contact sheet in
  `data/paired/uxo/projection_report.json` / `projection_check.jpg`
  and the W1–W3 log.
- GPU env ready for S0-lite on this box (RTX 5090 D v2 24 GB):
  `.venv-gpu` with torch 2.7.1+cu128 + source-built CPU-ops mmcv 2.1.0
  and an NMS CPU fallback (`aodet/cuda_compat.py`) — see
  `env/ENVIRONMENT.md`.  Pinned cu121 stack cannot run on Blackwell;
  the CPU/debug `.venv` is untouched.
- RSFusionDet repo still empty (route a watch ongoing).
