# Setup Guide

## Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU training; CPU works but is slow — see Hardware Requirements in the main README)
- ~2 GB disk space for datasets (N-MNIST ~30 MB, SHD ~1 GB)

## Installation

```bash
# 1. Clone the repo and enter the directory
git clone https://github.com/falakshah18/tsa_neuroips.git
cd tsa_neuroips

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dependency Notes

| Package | Version | Notes |
|---|---|---|
| `torch` | ≥2.0 | Install a CUDA-enabled build from pytorch.org if using GPU |
| `spikingjelly` | ≥0.0.0.0.14 | Spiking neuron primitives (`ParametricLIFNode`, `surrogate`) |
| `tonic` | ≥1.0, <2.0 | Neuromorphic dataset loader (N-MNIST, SHD) |
| `wandb` | ≥0.15 | Experiment tracking — set `use_wandb: false` in config to skip |
| `pyyaml`, `scipy` | — | Config parsing, statistical validation |

## GPU Setup (recommended)

```bash
# Check if CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

If `False`, install a CUDA-enabled torch build:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Datasets

`tonic` downloads datasets automatically on first run to `./data` in the repo root (not `~/.tonic/`). N-MNIST is ~30 MB; SHD is ~1 GB. Ensure you have network access on the first run for whichever dataset you use — after that, it's cached locally and no further downloads happen.

## Verifying the Install

```bash
python test_main.py
```

This checks imports and runs the full pytest suite (18 tests) — no dataset download required, finishes in a few seconds. Expected output:
```
✅ All imports successful
   torch: 2.x.x
   spikingjelly: installed
   numpy: 1.x.x
...
18 passed
✅ ALL TESTS PASSED
```

Once that passes, confirm real data flows through the pipeline correctly with a genuine (but fast) end-to-end run:
```bash
python main.py --mode baseline --algorithm ttfs --dataset shd --quick --n_seeds 1 --epochs 1 --device cpu
```
This downloads SHD on first run and trains one epoch of one baseline — a good way to confirm the environment is fully working before committing to a full multi-hour run.

## WandB (optional)

If `use_wandb: true` in your config:
```bash
wandb login   # paste your API key from wandb.ai/authorize
```

To run without WandB, set `use_wandb: false` in `configs/tsa_config.yaml` (already the default).

## Common Setup Issues

- **Slow / apparently frozen training on CPU** — training now shows a live progress bar with running loss per batch. If you don't see one, confirm `tqdm` installed correctly (`pip show tqdm`); the trainer falls back to silent (but still working) mode without it.
- **Process killed with no error message** — likely an out-of-memory kill from the OS, not a code bug, especially for TSA's default config (`embed_dim=256, depth=4`) on CPU. See Hardware Requirements in the main README.
- **`pin_memory` warning on CPU** — harmless; `pin_memory=True` is a no-op without a CUDA device.
