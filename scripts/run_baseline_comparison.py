# scripts/run_baseline_comparison.py
"""
Entry point to run the full baseline comparison.
This is the main UGRP experiment script.

Usage:
    # Full comparison (all algorithms, all datasets)
    python scripts/run_baseline_comparison.py

    # Single dataset
    python scripts/run_baseline_comparison.py --dataset nmnist

    # Single algorithm
    python scripts/run_baseline_comparison.py --algorithm surrogate_gradient

    # Quick test (1 seed, 5 epochs)
    python scripts/run_baseline_comparison.py --quick
"""

import argparse
import sys
import json
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from experiments.baseline_comparison import BaselineComparison


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run neuromorphic algorithm comparison'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['nmnist', 'shd'],
        choices=['nmnist', 'shd'],
        help='Datasets to evaluate on'
    )

    parser.add_argument(
        '--algorithms',
        nargs='+',
        default=[
            'Surrogate_Gradient',
            'ANN_to_SNN',
            'Supervised_STDP',
            'E_prop',
            'TTFS',
            'TSA_Ours',
        ],
        help='Algorithms to compare'
    )

    parser.add_argument(
        '--n_seeds',
        type=int,
        default=3,
        help='Number of random seeds (default: 3, paper uses 10)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs per run'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./baseline_comparison_results',
        help='Directory to save results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: 1 seed, 5 epochs, nmnist only'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to existing results JSON to resume from'
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# Environment Check
# ─────────────────────────────────────────────

def check_environment():
    """
    Verify all dependencies are installed
    before starting long experiments.
    """
    print("\n Checking environment...")

    checks = {
        'torch': 'import torch',
        'spikingjelly': 'from spikingjelly.activation_based import neuron',
        'tonic': 'import tonic',
        'numpy': 'import numpy',
        'scipy': 'import scipy',
        'sklearn': 'import sklearn',
        'yaml': 'import yaml',
        'wandb': 'import wandb',
    }

    all_ok = True
    for pkg, import_str in checks.items():
        try:
            exec(import_str)
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} — install with: pip install {pkg}")
            if pkg not in ['wandb']:  # wandb optional
                all_ok = False

    # Check CUDA
    if torch.cuda.is_available():
        print(f"  ✅ CUDA: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("  ⚠️  CUDA not available — using CPU (slow)")

    if not all_ok:
        print("\n❌ Missing dependencies. Run:")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    print("  ✅ Environment OK\n")


# ─────────────────────────────────────────────
# Resume Support
# ─────────────────────────────────────────────

def load_existing_results(resume_path: str) -> dict:
    """Load existing results to resume from."""
    with open(resume_path) as f:
        results = json.load(f)
    print(f"Resuming from: {resume_path}")
    print(f"Existing results: {list(results.keys())}")
    return results


def get_remaining_runs(
    existing_results: dict,
    datasets: list,
    algorithms: list,
    n_seeds: int,
) -> list:
    """
    Determine which (dataset, algorithm, seed)
    combinations still need to run.
    """
    remaining = []

    for dataset in datasets:
        for algo in algorithms:
            existing_seeds = len(
                existing_results
                .get(dataset, {})
                .get(algo, [])
            )
            for seed in range(existing_seeds, n_seeds):
                remaining.append((dataset, algo, seed))

    return remaining


# ─────────────────────────────────────────────
# Summary Printer
# ─────────────────────────────────────────────

def print_final_summary(results: dict):
    """
    Print clean final summary table.
    This is what you show Prof. Lakhlani.
    """
    print("\n" + "=" * 70)
    print("FINAL RESULTS: NEUROMORPHIC ALGORITHM COMPARISON")
    print("UGRP — Review and Evaluation")
    print("=" * 70)

    import numpy as np

    datasets = [k for k in results.keys()
                if k != 'bio_plausibility']
    bio_scores = results.get('bio_plausibility', {})

    for dataset in datasets:
        print(f"\n{'─'*70}")
        print(f"Dataset: {dataset.upper()}")
        print(f"{'─'*70}")
        print(
            f"{'Algorithm':<25} "
            f"{'Accuracy':<15} "
            f"{'Energy(uJ)':<15} "
            f"{'Bio Score':<12} "
            f"{'Time(min)':<10}"
        )
        print("─" * 70)

        algo_results = results.get(dataset, {})

        # Sort by accuracy
        summary = []
        for algo, seed_results in algo_results.items():
            if not isinstance(seed_results, list):
                continue
            valid = [r for r in seed_results if 'error' not in r]
            if not valid:
                continue

            mean_acc = np.mean([r['test_acc'] for r in valid])
            std_acc = np.std([r['test_acc'] for r in valid])
            mean_energy = np.mean([r.get('test_energy', 0) for r in valid])
            mean_time = np.mean([r.get('train_time', 0) for r in valid])
            bio = bio_scores.get(algo, {}).get('total_score', '-')

            summary.append((
                algo, mean_acc, std_acc,
                mean_energy, bio, mean_time
            ))

        summary.sort(key=lambda x: x[1], reverse=True)

        for algo, acc, std, energy, bio, time in summary:
            marker = " ←OURS" if algo == 'TSA_Ours' else ""
            print(
                f"{algo:<25} "
                f"{acc*100:.2f}±{std*100:.2f}%  "
                f"{energy:<15.4f} "
                f"{bio}/5         "
                f"{time:.1f}{marker}"
            )

    print("\n" + "=" * 70)
    print("Biological Plausibility Breakdown:")
    print("─" * 70)
    print(
        f"{'Algorithm':<25} "
        f"{'Local':<8} "
        f"{'Spike':<8} "
        f"{'Temporal':<10} "
        f"{'NoWT':<8} "
        f"{'Online':<8} "
        f"{'Total':<8}"
    )
    print("─" * 70)

    for algo, score_dict in bio_scores.items():
        criteria = score_dict.get('criteria', {})
        total = score_dict.get('total_score', 0)

        def yn(v): return "✅" if v else "❌"

        print(
            f"{algo:<25} "
            f"{yn(criteria.get('local_learning_rule')):<8} "
            f"{yn(criteria.get('spike_based_communication')):<8} "
            f"{yn(criteria.get('temporal_dynamics')):<10} "
            f"{yn(criteria.get('no_weight_transport')):<8} "
            f"{yn(criteria.get('online_learning')):<8} "
            f"{total}/5"
        )

    print("=" * 70)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Quick mode override -- only affects seeds/epochs. Datasets/algorithms
    # come from whatever was explicitly passed via --datasets/--algorithms
    # (main.py already computes the right dataset list before calling in).
    if args.quick:
        print(f"⚡ QUICK MODE: 1 seed, 5 epochs, datasets={args.datasets}")
        args.n_seeds = 1
        args.epochs = 5

    print("\n" + "=" * 60)
    print("TSA NEUROIPS — BASELINE COMPARISON")
    print("UGRP: Review and Evaluation of")
    print("Neuromorphic Computing Algorithms")
    print("=" * 60)
    print(f"Datasets:   {args.datasets}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Seeds:      {args.n_seeds}")
    print(f"Epochs:     {args.epochs}")
    print(f"Device:     {args.device}")
    print(f"Save dir:   {args.save_dir}")

    # Check environment
    check_environment()

    # Handle resume
    existing_results = None
    if args.resume:
        existing_results = load_existing_results(args.resume)

    # Run comparison
    comparison = BaselineComparison(
        datasets=args.datasets,
        algorithms=args.algorithms,
        n_seeds=args.n_seeds,
        epochs=args.epochs,
        save_dir=args.save_dir,
        device=args.device,
        quick=args.quick,
    )

    # Inject existing results if resuming
    if existing_results:
        comparison.results = existing_results

    results = comparison.run()
    report = comparison.generate_report()

    # Print summary
    print_final_summary(results)

    # Save final summary
    save_dir = Path(args.save_dir)
    with open(save_dir / 'final_summary.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✅ All results saved to {args.save_dir}/")
    print(f"   full_comparison.json — raw seed results")
    print(f"   ugrp_report.json     — aggregated report")
    print(f"   final_summary.json   — summary for paper")


if __name__ == '__main__':
    main()