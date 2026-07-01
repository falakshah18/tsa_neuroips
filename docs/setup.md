# Setup Guide

## Requirements

- Python 3.9+
- CUDA 11.8+ (for GPU training; CPU works but is slow)
- ~4 GB disk space for datasets

## Installation

```bash
# 1. Clone the repo and enter the directory
git clone <repo-url>
cd tsa-neuroips

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dependency Notes

| Package | Version | Notes |
|---|---|---|
| `torch` | ≥2.0 | Install CUDA-enabled build from pytorch.org if using GPU |
| `spikingjelly` | ≥0.0.0.0.14 | Spiking neuron primitives (`ParametricLIFNode`, `surrogate`) |
| `tonic` | ≥1.0, <2.0 | Neuromorphic dataset loader (N-MNIST, SHD) |
| `wandb` | ≥0.15 | Experiment tracking — set `use_wandb: false` in config to skip |

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

`tonic` downloads datasets automatically on first run to `~/.tonic/` (or the path set in the config). N-MNIST is ~30 MB; SHD is ~1 GB. Ensure you have network access on the first run.

## Verifying the Install

```bash
python test_main.py
```

Expected output:
```
✅ All imports successful
   torch: 2.x.x
   spikingjelly: 0.0.0.0.14
   numpy: 1.x.x
...
✅ ALL TESTS PASSED
```

## WandB (optional)

If `use_wandb: true` in your config:
```bash
wandb login   # paste your API key from wandb.ai/authorize
```

To run without WandB, set `use_wandb: false` in `configs/tsa_config.yaml` (already the default).
