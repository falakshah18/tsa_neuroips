"""
Unified experiment runner — runs ALL algorithms × ALL datasets × N seeds.

Usage:
    python scripts/run_all_experiments.py                          # default: all algs, all datasets, 5 seeds, 300 epochs
    python scripts/run_all_experiments.py --datasets nmnist        # N-MNIST only
    python scripts/run_all_experiments.py --n_seeds 1 --epochs 2   # quick smoke test
    python scripts/run_all_experiments.py --resume                 # resume from last checkpoint
    python scripts/run_all_experiments.py --skip_existing          # skip completed runs
    python scripts/run_all_experiments.py --generate_outputs       # only regenerate tables/plots from existing results

Saves:
    results/all_experiments/{dataset}/{algorithm}/seed{N}/result.json
    results/all_experiments/{dataset}/{algorithm}/seed{N}/checkpoint.pth
    results/all_experiments/full_results.json
    results/all_experiments/summary.json
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.dataset_loaders import get_all_dataset_loaders, DATASET_CONFIGS
from training.trainer_v2 import AdvancedTrainer
from utils import set_seed


ALL_ALGORITHMS = [
    'surrogate_gradient',
    'ann_to_snn',
    'stdp',
    'eprop',
    'ttfs',
    'tsa',
]

ALL_DATASETS = ['nmnist', 'shd', 'dvs_gesture', 'cifar10_dvs']

CONFIG_DIR = Path(__file__).resolve().parent.parent / 'configs'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'all_experiments'


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_algo_config(algorithm):
    file_map = {
        'surrogate_gradient': 'surrogate_grad_config.yaml',
        'ann_to_snn': 'ann_to_snn_config.yaml',
        'stdp': 'stdp_config.yaml',
        'eprop': 'eprop_config.yaml',
        'ttfs': 'ttfs_config.yaml',
        'tsa': 'tsa_config.yaml',
    }
    path = CONFIG_DIR / file_map.get(algorithm, f'{algorithm}_config.yaml')
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


def get_training_config(algorithm, dataset, seed, epochs, save_dir):
    algo_cfg = load_algo_config(algorithm)
    train_cfg = algo_cfg.get('training', {})
    return {
        'epochs': epochs,
        'lr': train_cfg.get('lr', 0.001),
        'weight_decay': train_cfg.get('weight_decay', 0.05),
        'warmup_epochs': train_cfg.get('warmup_epochs', 10),
        'patience': train_cfg.get('patience', 30),
        'mixed_precision': train_cfg.get('mixed_precision', True),
        'gradient_accumulation_steps': train_cfg.get('gradient_accumulation_steps', 1),
        'spike_reg': train_cfg.get('spike_reg', 0.001),
        'label_smoothing': train_cfg.get('label_smoothing', 0.0),
        'grad_clip': train_cfg.get('grad_clip', 1.0),
        'log_dir': str(save_dir / 'logs'),
        'checkpoint_dir': str(save_dir / 'checkpoints'),
        'project_name': f'{algorithm.upper()}_NEUROIPS',
        'run_name': f'{algorithm}_{dataset}_seed{seed}',
        'save_every': 50,
    }


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def create_model(algorithm, dataset, device):
    ds_cfg = DATASET_CONFIGS[dataset]
    algo_cfg = load_algo_config(algorithm)

    if algorithm == 'tsa':
        from models.tst_v2 import TemporalSpikingTransformer
        m = algo_cfg.get('model', {})
        return TemporalSpikingTransformer(
            img_size=ds_cfg['img_size'],
            patch_size=ds_cfg['patch_size'],
            in_channels=ds_cfg['in_channels'],
            num_classes=ds_cfg['num_classes'],
            embed_dim=m.get('embed_dim', 256),
            depth=m.get('depth', 4),
            num_heads=m.get('num_heads', 8),
            mlp_ratio=m.get('mlp_ratio', 4.0),
            init_tau=m.get('init_tau', 2.0),
        )

    elif algorithm == 'surrogate_gradient':
        from baselines.surrogate_gradient import get_surrogate_grad_model
        return get_surrogate_grad_model(dataset, algo_cfg)

    elif algorithm == 'stdp':
        from baselines.stdp import get_stdp_model
        return get_stdp_model(dataset, algo_cfg)

    elif algorithm == 'eprop':
        from baselines.eprop import get_eprop_model
        return get_eprop_model(dataset, algo_cfg)

    elif algorithm == 'ttfs':
        from baselines.temporal_coding import get_ttfs_model
        return get_ttfs_model(dataset, algo_cfg)

    elif algorithm == 'ann_to_snn':
        from baselines.ann_to_snn import get_ann_to_snn_model
        ann, snn_class, snn_kwargs = get_ann_to_snn_model(dataset, algo_cfg)
        return ann

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")



# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def get_loaders(dataset, batch_size=32, quick=False):
    cfg = DATASET_CONFIGS[dataset]
    return get_all_dataset_loaders(
        dataset_name=dataset,
        batch_size=batch_size,
        n_time_bins=cfg['n_time_bins'],
        quick=quick,
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(algorithm, dataset, seed, epochs, save_dir, device, quick=False):
    run_dir = save_dir / dataset / algorithm / f'seed{seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / 'result.json'

    if result_path.exists():
        print(f"    [SKIP] {result_path} already exists")
        with open(result_path) as f:
            return json.load(f)

    print(f"  [{algorithm}] [{dataset}] seed={seed}", flush=True)

    set_seed(seed)

    try:
        train_loader, val_loader, test_loader = get_loaders(
            dataset,
            batch_size=32,
            quick=quick,
        )

        model = create_model(algorithm, dataset, device)
        model = model.to(device)

        train_config = get_training_config(algorithm, dataset, seed, epochs, run_dir)

        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=train_config,
            device=device,
            use_wandb=False,
        )

        t0 = time.time()
        result = trainer.train()
        elapsed = time.time() - t0

        output = {
            'algorithm': algorithm,
            'dataset': dataset,
            'seed': seed,
            'test_acc': result['test_metrics']['acc'],
            'test_energy_uJ': result['test_metrics'].get('avg_energy_uJ', 0.0),
            'test_loss': result['test_metrics'].get('loss', 0.0),
            'best_val_acc': result['best_val_acc'],
            'train_time_minutes': elapsed / 60.0,
            'epochs_trained': epochs,
            'error': False,
        }

        with open(result_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"    acc={output['test_acc']*100:.2f}%  "
              f"val={output['best_val_acc']*100:.2f}%  "
              f"energy={output['test_energy_uJ']:.4f} uJ  "
              f"time={output['train_time_minutes']:.1f}min")

        return output

    except Exception as e:
        import traceback
        traceback.print_exc()
        output = {
            'algorithm': algorithm,
            'dataset': dataset,
            'seed': seed,
            'error': str(e),
            'test_acc': 0.0,
            'test_energy_uJ': 0.0,
            'best_val_acc': 0.0,
            'train_time_minutes': 0.0,
        }
        with open(result_path, 'w') as f:
            json.dump(output, f, indent=2)
        return output


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------
def run_all(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    datasets = args.datasets or ALL_DATASETS
    algorithms = args.algorithms or ALL_ALGORITHMS
    n_seeds = args.n_seeds
    epochs = args.epochs
    device = args.device
    quick = args.quick

    total_runs = len(datasets) * len(algorithms) * n_seeds
    estimated_hours = total_runs * (1.5 if not quick else 0.01)
    print("=" * 70)
    print("UNIFIED EXPERIMENT RUNNER")
    print("=" * 70)
    print(f"  Datasets:    {datasets}")
    print(f"  Algorithms:  {algorithms}")
    print(f"  Seeds:       {n_seeds}")
    print(f"  Epochs:      {epochs}")
    print(f"  Device:      {device}")
    print(f"  Quick mode:  {quick}")
    print(f"  Total runs:  {total_runs}")
    print(f"  Est. time:   ~{estimated_hours:.0f}h (rough estimate)")
    print(f"  Results dir: {save_dir}")
    print("=" * 70)

    all_results = {}
    run_count = 0
    start_time = time.time()

    for dataset in datasets:
        all_results[dataset] = {}
        ds_start = time.time()

        for algorithm in algorithms:
            all_results[dataset][algorithm] = []

            for seed in range(n_seeds):
                run_count += 1
                print(f"\n[{run_count}/{total_runs}] ", end="")

                result = run_single(
                    algorithm=algorithm,
                    dataset=dataset,
                    seed=seed,
                    epochs=epochs,
                    save_dir=save_dir,
                    device=device,
                    quick=quick,
                )
                all_results[dataset][algorithm].append(result)

        ds_elapsed = time.time() - ds_start
        print(f"\n  Dataset {dataset} done in {timedelta(seconds=int(ds_elapsed))}")

    # Save full results
    full_path = save_dir / 'full_results.json'
    with open(full_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate summary
    summary = generate_summary(all_results, datasets, algorithms)
    with open(save_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    total_elapsed = time.time() - start_time
    print_summary(summary, datasets, algorithms, total_elapsed)

    return all_results


# ---------------------------------------------------------------------------
# Summary & reporting
# ---------------------------------------------------------------------------
def generate_summary(all_results, datasets, algorithms):
    summary = {}
    for dataset in datasets:
        summary[dataset] = {}
        for algo in algorithms:
            runs = all_results.get(dataset, {}).get(algo, [])
            valid = [r for r in runs if not r.get('error', False)]
            if valid:
                accs = [r['test_acc'] for r in valid]
                energies = [r.get('test_energy_uJ', 0) for r in valid]
                times = [r.get('train_time_minutes', 0) for r in valid]
                summary[dataset][algo] = {
                    'mean_acc': float(np.mean(accs)),
                    'std_acc': float(np.std(accs)),
                    'mean_energy_uJ': float(np.mean(energies)),
                    'std_energy_uJ': float(np.std(energies)),
                    'mean_time_min': float(np.mean(times)),
                    'n_seeds': len(valid),
                    'n_failed': len(runs) - len(valid),
                }
            else:
                summary[dataset][algo] = {
                    'mean_acc': 0.0, 'std_acc': 0.0,
                    'mean_energy_uJ': 0.0, 'std_energy_uJ': 0.0,
                    'mean_time_min': 0.0,
                    'n_seeds': 0, 'n_failed': len(runs),
                }
    return summary


def print_summary(summary, datasets, algorithms, total_elapsed):
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    for dataset in datasets:
        print(f"\n  {dataset.upper()}")
        print(f"  {'Algorithm':<25} {'Acc (%)':<15} {'Energy (uJ)':<15} {'Time (min)':<12}")
        print(f"  {'-'*67}")
        for algo in algorithms:
            s = summary.get(dataset, {}).get(algo, {})
            if s.get('n_seeds', 0) > 0:
                print(f"  {algo:<25} "
                      f"{s['mean_acc']*100:>5.2f} ± {s['std_acc']*100:<5.2f}  "
                      f"{s['mean_energy_uJ']:>7.4f} ± {s['std_energy_uJ']:<5.4f}  "
                      f"{s['mean_time_min']:>8.1f}")
            else:
                print(f"  {algo:<25} {'FAILED':>15}")
    print(f"\n  Total time: {timedelta(seconds=int(total_elapsed))}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Output generation (tables + plots)
# ---------------------------------------------------------------------------
def generate_outputs(args):
    save_dir = Path(args.save_dir)
    results_path = save_dir / 'full_results.json'

    if not results_path.exists():
        print(f"No results found at {results_path}. Run experiments first.")
        return

    with open(results_path) as f:
        all_results = json.load(f)

    print("Generating LaTeX tables...")
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from generate_comparison_table import (
            generate_algorithm_comparison,
            generate_bio_plausibility_table,
        )
        paper_tables = Path(__file__).resolve().parent.parent / 'paper' / 'tables'
        paper_tables.mkdir(parents=True, exist_ok=True)

        datasets = list(all_results.keys())
        generate_algorithm_comparison(all_results, datasets, paper_tables)
        generate_bio_plausibility_table(paper_tables)
        print(f"  Tables saved to {paper_tables}")
    except Exception as e:
        print(f"  Table generation failed: {e}")

    print("Generating plots...")
    try:
        from generate_plots import (
            plot_algorithm_comparison,
            plot_energy_comparison,
        )
        paper_figs = Path(__file__).resolve().parent.parent / 'paper' / 'figures'
        paper_figs.mkdir(parents=True, exist_ok=True)

        datasets_list = list(all_results.keys())
        plot_algorithm_comparison(all_results, datasets_list, paper_figs)
        plot_energy_comparison(all_results, datasets_list, paper_figs)
        print(f"  Plots saved to {paper_figs}")
    except Exception as e:
        print(f"  Plot generation failed: {e}")

    print("Output generation complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run ALL experiments: algorithms x datasets x seeds'
    )
    parser.add_argument('--datasets', nargs='+', default=None,
                        choices=ALL_DATASETS,
                        help='Datasets to run (default: all 4)')
    parser.add_argument('--algorithms', nargs='+', default=None,
                        choices=ALL_ALGORITHMS,
                        help='Algorithms to run (default: all 6)')
    parser.add_argument('--n_seeds', type=int, default=5,
                        help='Number of random seeds (default: 5)')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Training epochs (default: 300)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (default: cuda)')
    parser.add_argument('--save_dir', type=str,
                        default=str(RESULTS_DIR),
                        help='Results directory')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode (smaller data, fewer epochs)')
    parser.add_argument('--generate_outputs', action='store_true',
                        help='Only generate tables/plots from existing results')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip runs that already have result.json')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.generate_outputs:
        generate_outputs(args)
        return

    if not torch.cuda.is_available() and args.device == 'cuda':
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    run_all(args)

    print("\nDone. To regenerate tables/plots:")
    print("  python scripts/run_all_experiments.py --generate_outputs")


if __name__ == '__main__':
    main()
