#!/usr/bin/env python
# main.py
"""
Main entry point for TSA NeuroIPS project.

Usage:
    # Run baseline comparison
    python main.py --mode baseline

    # Run TSA experiments
    python main.py --mode tsa

    # Run all experiments
    python main.py --mode all

    # Quick test
    python main.py --mode all --quick
"""

import argparse
import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description='TSA NeuroIPS - Main Entry Point'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['baseline', 'tsa', 'ablations', 'hardware', 'all'],
        help='Which experiments to run'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        choices=['nmnist', 'shd'],
        help='Specific dataset to run on'
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default=None,
        help='Specific algorithm to run (for baseline mode)'
    )

    parser.add_argument(
        '--n_seeds',
        type=int,
        default=3,
        help='Number of random seeds'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use'
    )

    parser.add_argument(
        '--generate_tables',
        action='store_true',
        help='Generate LaTeX tables after experiments'
    )

    parser.add_argument(
        '--generate_figures',
        action='store_true',
        help='Generate paper figures after experiments'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("TSA NEUROIPS — MAIN ENTRY POINT")
    print("=" * 70)
    print(f"Mode:     {args.mode}")
    print(f"Dataset:  {args.dataset or 'all'}")
    print(f"Seeds:    {args.n_seeds}")
    print(f"Epochs:   {args.epochs}")
    print(f"Device:   {args.device}")
    print(f"Quick:    {args.quick}")

    # Set datasets
    datasets = [args.dataset] if args.dataset else ['nmnist', 'shd']

    # Override for quick mode -- but never clobber an explicitly-requested
    # dataset. --quick means "fast settings" (1 seed, 5 epochs), not
    # "always nmnist, ignore whatever --dataset was passed".
    if args.quick:
        if not args.dataset:
            datasets = ['nmnist']
        args.n_seeds = 1
        args.epochs = 5
        print("\nQUICK MODE ENABLED")

    # Run baseline comparison
    if args.mode in ['baseline', 'all']:
        print("\n" + "-" * 70)
        print("RUNNING BASELINE COMPARISON")
        print("-" * 70)

        from scripts.run_baseline_comparison import main as baseline_main

        # Use argparse to pass arguments
        sys.argv = [
            'run_baseline_comparison.py',
            '--datasets'] + datasets + [
            '--n_seeds', str(args.n_seeds),
            '--epochs', str(args.epochs),
            '--device', args.device,
        ]

        if args.algorithm:
            sys.argv.extend(['--algorithms', args.algorithm])

        if args.quick:
            sys.argv.append('--quick')

        baseline_main()

    # Run TSA experiments
    if args.mode in ['tsa', 'all']:
        print("\n" + "-" * 70)
        print("RUNNING TSA EXPERIMENTS")
        print("-" * 70)

        from scripts.run_tsa_experiments import main as tsa_main

        sys.argv = [
            'run_tsa_experiments.py',
            '--datasets'] + datasets + [
            '--n_seeds', str(args.n_seeds),
            '--epochs', str(args.epochs),
            '--device', args.device,
        ]

        if args.mode == 'tsa':
            sys.argv.extend(['--mode', 'train'])

        if args.quick:
            sys.argv.append('--quick')

        tsa_main()

    # Generate tables
    if args.generate_tables:
        print("\n" + "-" * 70)
        print("GENERATING LATEX TABLES")
        print("-" * 70)

        from scripts.generate_comparison_table import main as table_main
        table_main()

    # Generate figures
    if args.generate_figures:
        print("\n" + "-" * 70)
        print("GENERATING PAPER FIGURES")
        print("-" * 70)

        from scripts.generate_plots import main as figure_main
        figure_main()

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()