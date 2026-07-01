# TSA — Temporal Spiking Attention

A biologically-plausible spiking neural network with learnable temporal attention, targeting NeurIPS submission on neuromorphic computing.

## Architecture

- **LearnableTSA**: Temporal Spike Attention with per-head learnable decay (`tau`), threshold, and temperature parameters
- **TemporalSpikingTransformer (TST)**: Full transformer using TSA blocks + spiking MLP, operating on event-based neuromorphic data
- Datasets: **N-MNIST** (34×34, 2-channel) and **SHD** (700 channels, audio)

## Setup

See `docs/setup.md` for full instructions.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Sanity check (imports + unit tests)
python test_main.py

# Quick run — 1 seed, 5 epochs, N-MNIST only
python main.py --mode all --quick

# Train TSA on N-MNIST
python main.py --mode tsa --dataset nmnist --epochs 100

# Full paper experiments
python main.py --mode all --n_seeds 3 --epochs 300 --generate_tables --generate_figures
```

## Project Structure

```
models/          # TemporalSpikingTransformer, LearnableTSA
training/        # AdvancedTrainer (mixed precision, cosine LR, early stopping)
baselines/       # ANN-to-SNN, eProp, STDP, surrogate gradient, TTFS
experiments/     # Ablations, benchmarks, statistical validation
hardware/        # Loihi 2 deployment & energy estimation
configs/         # Per-algorithm YAML configs
scripts/         # run_tsa_experiments.py, run_baseline_comparison.py, plots, tables
tests/           # pytest suite
```

## Running Individual Scripts

```bash
# Baseline comparison only
python scripts/run_baseline_comparison.py --datasets nmnist shd --quick

# TSA experiments only
python scripts/run_tsa_experiments.py --mode train --dataset nmnist

# Generate LaTeX tables
python scripts/generate_comparison_table.py

# Generate paper figures
python scripts/generate_plots.py
```
