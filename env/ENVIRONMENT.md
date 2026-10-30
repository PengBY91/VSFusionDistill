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
| CUDA       | ≥11.8      | none (no GPU visible)    | TBD             |

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
