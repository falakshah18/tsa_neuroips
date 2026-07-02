# TSA — Temporal Spiking Attention

A biologically-plausible spiking neural network with learnable temporal attention, targeting a NeurIPS submission on neuromorphic computing.

TSA replaces the standard softmax attention mechanism used in transformers with a spiking-neuron-based attention rule: instead of computing continuous attention weights, per-head leaky integrate-and-fire (LIF) neurons build up membrane potential from query/key coincidence over time and only "attend" when they spike. The neurons' own decay rate (`tau`), firing threshold, and temperature are learned during training rather than fixed.

## Status

Research code under active development for a NeurIPS submission. The core architecture, training pipeline, and all five baselines have been verified end-to-end (correct shapes, dtypes, and gradient flow, using data matching tonic's real output exactly) on both datasets below. Benchmark numbers are not yet included here — they'll be added once full multi-seed runs (300 epochs, 3+ seeds, per `configs/tsa_config.yaml`) are complete against the real downloaded datasets, rather than reporting placeholder or short-run numbers as if they were final results.

**Known limitation:** TSA's patch embedding is a genuine 2D `Conv2d` (`num_patches = (img_size // patch_size) ** 2`), which fits N-MNIST's real 2D event-camera structure but not SHD's 1D, 700-channel spike train. TSA currently only trains on **N-MNIST**. All five baselines (surrogate gradient, ANN-to-SNN, STDP, e-prop, TTFS) support both N-MNIST and SHD. Extending TSA to SHD needs a dedicated 1D patch-embedding variant, which hasn't been built yet.

## Architecture

- **LearnableTSA**: Temporal Spike Attention with per-head learnable decay (`tau`), threshold, and temperature parameters
- **TemporalSpikingTransformer (TST)**: Full transformer using TSA blocks + spiking MLP, operating on event-based neuromorphic data
- Datasets: **N-MNIST** (34×34, 2-channel, TSA + all baselines) and **SHD** (700 channels, audio, baselines only — see Known Limitation above)

## Setup

See [`docs/setup.md`](docs/setup.md) for full instructions, including GPU setup and dependency notes.

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Hardware Requirements

| | Minimum | Recommended |
|---|---|---|
| RAM | 8 GB | 16 GB+ |
| GPU | none (CPU works) | CUDA-capable, 8 GB+ VRAM |
| Disk | ~2 GB (N-MNIST + SHD) | — |

CPU training works but is slow, especially for TSA (`embed_dim=256, depth=4` by default — this is a full transformer running per-timestep over up to 20–100 steps). If you hit an unexplained kill with no traceback on CPU, it's very likely an out-of-memory kill from the OS, not a code error — try a smaller `embed_dim`/`depth` in `configs/tsa_config.yaml` or move to GPU.

## Quick Start

```bash
# Sanity check (imports + unit tests, ~5 seconds, no dataset needed)
python test_main.py

# Quick run — 1 seed, 5 epochs, fast settings
# (defaults to N-MNIST if --dataset isn't given; add --algorithm to restrict to one baseline)
python main.py --mode all --quick

# Run one baseline algorithm on one dataset
python main.py --mode baseline --algorithm ttfs --dataset shd --quick

# Train TSA on N-MNIST (SHD not currently supported for TSA — see Known Limitation)
python main.py --mode tsa --dataset nmnist --epochs 100

# Full paper experiments
python main.py --mode all --n_seeds 3 --epochs 300 --generate_tables --generate_figures
```

The first run of any dataset downloads it automatically via `tonic` to `./data` (N-MNIST ~30 MB, SHD ~1 GB) — this needs network access once, then it's cached locally.

## Project Structure

```
models/          # TemporalSpikingTransformer, LearnableTSA
training/        # AdvancedTrainer (mixed precision, cosine LR, early stopping, checkpointing)
baselines/       # ANN-to-SNN, e-prop, STDP, surrogate gradient, TTFS
experiments/     # Baseline comparison, ablations, statistical validation
hardware/        # Loihi 2 deployment & energy estimation
theory/          # Convergence/energy/complexity proofs (generates the *.pdf figures at repo root)
configs/         # Per-algorithm YAML configs
scripts/         # run_tsa_experiments.py, run_baseline_comparison.py, plots, tables
tests/           # pytest suite (18 tests — model, baseline, and trainer sanity checks)
paper/           # LaTeX source
```

## Running Individual Scripts

```bash
# Baseline comparison only, both datasets
python scripts/run_baseline_comparison.py --datasets nmnist shd --quick

# Restrict to specific algorithms/datasets
python scripts/run_baseline_comparison.py --datasets shd --algorithms ttfs stdp

# TSA experiments only
python scripts/run_tsa_experiments.py --mode train --dataset nmnist

# Generate LaTeX tables
python scripts/generate_comparison_table.py

# Generate paper figures
python scripts/generate_plots.py
```

## Troubleshooting

- **`RuntimeError: expected scalar type Short but found Float`** — fixed as of the current version; if you see this, make sure you've pulled the latest `training/trainer_v2.py` (tonic's event-frame data loads as `int16`, which the trainer now casts to float automatically).
- **Process silently killed, no traceback, on CPU** — almost always an OS-level out-of-memory kill, not a crash. See Hardware Requirements above.
- **`--algorithm`/`--dataset` seem ignored** — fixed as of the current version. If you still see this, confirm you're on the latest commit (`git log --oneline -1`).

## Contributing

This is an active research codebase for a specific paper submission rather than a general-purpose library, so it isn't currently set up for external contributions. Feel free to open an issue if you spot a bug.

## Citation

A citation entry will be added once the paper is public.

## License

See [`LICENSE`](LICENSE).
