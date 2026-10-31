# Environment record

Pinned versions (both machines, from the setup doc §4.1):

| component  | pin        | debug machine (this box) | training machine |
| ---------- | ---------- | ------------------------ | ---------------- |
| Python     | 3.10       | 3.10.12 (system venv)    | TBD             |
| PyTorch    | 2.1.2      | 2.1.2+cpu                | 2.1.2+cu121     |
| mmcv       | 2.1.0      | 2.1.0 (cpu wheel)        | 2.1.0 (cu121 wheel) |
| mmengine   | 0.10.7     | 0.10.7                   | TBD             |
| mmdet      | 3.3.0      | 3.3.0                    | TBD             |
| numpy      | 1.26.x     | 1.26.4                   | TBD             |
| CUDA       | ≥11.8      | none at setup; see GPU env below | TBD     |

## Debug machine — setup record (2026-10-30)

- WSL2 (Linux 6.18.54.1-microsoft-standard-WSL2), 32 cores, 62 GiB RAM,
  no `nvidia-smi` → CPU-only build installed via `env/install-debug-machine.sh`.
- Wheels: torch/torchvision CPU index (download.pytorch.org/whl/cpu);
  mmcv CPU wheel from download.openmmlab.com/mmcv/dist/cpu/torch2.1.0.
- Full freeze: `env/pip-freeze.txt` (committed).
- `python -c "import torch; torch.cuda.is_available()"` → False (expected).

## Training machine — placeholder

24 GB-class GPU required (setup doc §8).  On first setup, run the cu121
install (env/environment.yml), then append here: `nvidia-smi` output,
`pip freeze` diff vs the debug machine, and the mmdet YOLOX-S COCO
reproduction number (acceptance: within 1 AP of the official 40.6/0.5 box).

## GPU environment on this box — `.venv-gpu` (2026-10-31)

The debug machine's hardware changed: it now carries an RTX 5090 D v2,
24 455 MiB (Blackwell, sm_120), visible to WSL2 via `nvidia-smi` (the
2026-10-30 "no GPU visible" record above is outdated).  This meets the
24 GB training requirement, so an S0-lite-capable env was built here.

Why the pinned cu121 stack cannot be used on it (measured, not
assumed): `torch 2.1.2+cu121` wheels contain only sm_50..sm_90 cubins
with no PTX fallback, so on sm_120 every kernel launch raises
`CUDA error: no kernel image is available for execution on the device`.
sm_120 needs cu128, i.e. torch >= 2.7.

Deviations from the pins (deliberate, documented here):

| component  | pin        | .venv-gpu                | why |
| ---------- | ---------- | ------------------------ | --- |
| PyTorch    | 2.1.2      | 2.7.1+cu128              | first line with native sm_120 kernels; 2048² matmul 0.07 s, conv fwd+bwd OK |
| torchvision| 0.16.2     | 0.22.1+cu128             | must match torch |
| mmcv       | 2.1.0 wheel| 2.1.0 source build, CPU-only ops | OpenMMLab ships no wheels past cu121/torch2.4; mmdet 3.3.0 requires mmcv<2.2.0; no nvcc on this box |
| setuptools | —          | <81                      | mmcv 2.1.0 setup.py imports `pkg_resources` (removed in setuptools 81) |

Everything else mirrors `env/pip-freeze.txt` (mmengine 0.10.7, mmdet
3.3.0, numpy 1.26.4, ...); full freeze: `env/pip-freeze-gpu.txt`.

mmcv build recipe (source tree kept at `~/mmcv-2.1.0-src`):

    pip install "setuptools<81" ninja
    CUDA_VISIBLE_DEVICES="" MMCV_WITH_OPS=1 \
      pip install ~/mmcv-2.1.0-src --no-build-isolation

`CUDA_VISIBLE_DEVICES=""` is essential: with the GPU visible,
`torch.cuda.is_available()` is True and mmcv's setup.py then takes the
`CUDAExtension` branch, which dies on the missing `CUDA_HOME`.  Hiding
the GPU lands it in the `CppExtension` (CPU-only ops) branch.

Consequence and mitigation: the compiled `_ext` has CPU kernels
only, so `mmcv.ops.nms` raises on CUDA tensors.  The training path uses
no compiled mmcv op (YOLOX losses / SimOTA are pure torch; our paired
transforms use `mmcv.transforms` only); inference-time NMS goes through
`aodet/cuda_compat.py::install_nms_cpu_fallback()` — self-detecting
(wired into `tools/test.py`; no-op wherever CUDA NMS already works).
Verified: imports clean, CPU `batched_nms` correct, shimmed GPU-tensor
NMS returns results on `cuda:0` with the same keep set.  Full test
suite on this env: see `doc/logs/2026-10-30-f2-codebase-w1-w3.md`.

Second torch-version flip handled: torch >= 2.6 defaults
`torch.load(weights_only=True)`; mmengine 0.10.7 passes no override and
its checkpoints embed `HistoryBuffer`/numpy objects in `meta`, so
`tools/test.py` loads local checkpoints through a scoped
`weights_only=False` wrapper (trusted own artifacts).  A future
`cfg.resume=True` on this env would need the same workaround; the
S0-lite runs train from scratch and are unaffected.

The pinned CPU/debug `.venv` is untouched and remains the reproducibility
baseline; the cu121 `env/environment.yml` stays for any true 24 GB
Ampere/Ada training machine where the pins work as-is.

Running S0-lite on this box (the two configs fit the 24 GB card in
parallel, ~4 GB B0 + ~9 GB cross-attn):

    export PATH="$PWD/.venv-gpu/bin:$PATH"
    RESULTS=results/results_b0.csv bash tools/run_seeds.sh \
        configs/f2/s0_b0_sonar_only_yoloxs.py 0 &
    RESULTS=results/results_xattn.csv bash tools/run_seeds.sh \
        configs/f2/s0_cross_attn_yoloxs.py 0 &

(Separate `RESULTS` files avoid the two collectors racing on
`results/results.csv`; merge afterwards.)  Rough wall-clock with both
running: B0 ~1.7 h, cross-attn ~3.5 h, plus the D0–D∞ scan.
