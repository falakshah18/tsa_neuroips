# scripts/run_tsa_experiments.py
"""
Run all TSA-specific experiments:
    1. Main TSA training on all datasets
    2. Ablation studies
    3. Hardware energy estimation
    4. Statistical validation vs baselines

Usage:
    # Full TSA experiments
    python scripts/run_tsa_experiments.py

    # Single dataset
    python scripts/run_tsa_experiments.py --dataset nmnist

    # Only ablations
    python scripts/run_tsa_experiments.py --mode ablations

    # Only benchmarks
    python scripts/run_tsa_experiments.py --mode benchmarks

    # Quick test
    python scripts/run_tsa_experiments.py --quick
"""

import argparse
import sys
import json
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from models.tst_v2 import TemporalSpikingTransformer
from training.trainer_v2 import AdvancedTrainer
from experiments.ablations import AblationFramework
from experiments.benchmarks import BaselineComparison
from experiments.statistical_validation import StatisticalValidator
from hardware.loihi2_deployment import LoihiSimulator
from tonic import datasets, transforms
from torch.utils.data import DataLoader, random_split
from spikingjelly.activation_based import functional


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run TSA experiments'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=[
            'all',
            'train',
            'ablations',
            'benchmarks',
            'hardware',
            'statistical',
        ],
        help='Which experiments to run'
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['nmnist', 'shd'],
        choices=['nmnist', 'shd'],
        help='Datasets to train on'
    )

    parser.add_argument(
        '--n_seeds',
        type=int,
        default=3,
        help='Random seeds for statistical validation'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=300,
        help='Training epochs'
    )

    parser.add_argument(
        '--ablation_epochs',
        type=int,
        default=100,
        help='Epochs for each ablation run'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device'
    )

    parser.add_argument(
        '--save_dir',
        type=str,
        default='./results',
        help='Results directory'
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints/tsa',
        help='Checkpoint directory'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test: 1 seed, 5 epochs'
    )

    parser.add_argument(
        '--load_checkpoint',
        type=str,
        default=None,
        help='Load pretrained checkpoint'
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# Config Loader
# ─────────────────────────────────────────────

def load_tsa_config() -> dict:
    """Load TSA config from yaml."""
    config_path = (
        Path(__file__).parent.parent /
        'configs' / 'tsa_config.yaml'
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Data Loaders
# ─────────────────────────────────────────────

def get_nmnist_loaders(
    batch_size: int = 32,
    n_time_bins: int = 20,
) -> tuple:
    """N-MNIST data loaders."""
    sensor_size = (34, 34, 2)
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=n_time_bins
        ),
    ])

    train_ds = datasets.NMNIST(
        save_to='./data', train=True,
        transform=transform
    )
    test_ds = datasets.NMNIST(
        save_to='./data', train=False,
        transform=transform
    )

    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_shd_loaders(
    batch_size: int = 32,
    n_time_bins: int = 100,
) -> tuple:
    """SHD data loaders."""
    sensor_size = (700, 1, 2)
    transform = transforms.Compose([
        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=n_time_bins
        ),
    ])

    train_ds = datasets.SHD(
        save_to='./data', train=True,
        transform=transform
    )
    test_ds = datasets.SHD(
        save_to='./data', train=False,
        transform=transform
    )

    train_size = int(0.9 * len(train_ds))
    val_size = len(train_ds) - train_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_loaders(dataset: str, config: dict) -> tuple:
    """Factory for data loaders."""
    if dataset == 'nmnist':
        return get_nmnist_loaders(
            batch_size=config.get('batch_size', 32),
            n_time_bins=config.get('T', 20),
        )
    elif dataset == 'shd':
        return get_shd_loaders(
            batch_size=config.get('batch_size', 32),
            n_time_bins=config.get('T', 100),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


# ─────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────

def get_tsa_model(dataset: str, config: dict) -> TemporalSpikingTransformer:
    """Create TSA model for given dataset."""
    ds_config = config.get(dataset, {})
    model_config = config.get('model', {})

    return TemporalSpikingTransformer(
        img_size=ds_config.get('img_size', 34),
        patch_size=ds_config.get('patch_size', 2),
        in_channels=ds_config.get('in_channels', 2),
        num_classes=ds_config.get('num_classes', 10),
        embed_dim=model_config.get('embed_dim', 256),
        depth=model_config.get('depth', 4),
        num_heads=model_config.get('num_heads', 8),
        mlp_ratio=model_config.get('mlp_ratio', 4.0),
        qkv_bias=model_config.get('qkv_bias', True),
        init_tau=model_config.get('init_tau', 2.0),
        drop_rate=model_config.get('drop_rate', 0.0),
        attn_drop_rate=model_config.get('attn_drop_rate', 0.0),
    )


# ─────────────────────────────────────────────
# Experiment 1: Main Training
# ─────────────────────────────────────────────

def run_training(
    datasets_list: List[str],
    n_seeds: int,
    epochs: int,
    config: dict,
    save_dir: Path,
    device: str,
    load_checkpoint: str = None,
) -> Dict:
    """
    Train TSA on all datasets with multiple seeds.
    Core experiment for paper Table 1.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: TSA MAIN TRAINING")
    print("=" * 60)

    all_results = {}

    for dataset in datasets_list:
        print(f"\nDataset: {dataset.upper()}")
        all_results[dataset] = []

        ds_config = config.get(dataset, {})
        train_config = config.get('training', {})

        # Get data loaders
        try:
            train_loader, val_loader, test_loader = get_loaders(
                dataset, ds_config
            )
        except Exception as e:
            print(f"  ⚠️  Could not load dataset '{dataset}': {e}")
            all_results[dataset].append({'error': str(e)})
            continue

        for seed in range(n_seeds):
            print(f"\n  Seed {seed + 1}/{n_seeds}")

            try:
                # Set seeds
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # Create model
                model = get_tsa_model(dataset, config)

                # Load checkpoint if provided
                if load_checkpoint:
                    ckpt = torch.load(load_checkpoint)
                    model.load_state_dict(ckpt['model_state_dict'])
                    print(f"  Loaded checkpoint: {load_checkpoint}")

                # Training config
                training_config = {
                    'project_name': 'TSA_NEUROIPS',
                    'run_name': f'TSA_{dataset}_seed{seed}',
                    'epochs': epochs,
                    'lr': train_config.get('lr', 1e-3),
                    'weight_decay': train_config.get('weight_decay', 0.05),
                    'warmup_epochs': train_config.get('warmup_epochs', 10),
                    'patience': train_config.get('patience', 30),
                    'mixed_precision': train_config.get('mixed_precision', True),
                    'gradient_accumulation_steps': train_config.get(
                        'gradient_accumulation_steps', 1
                    ),
                    'spike_reg': train_config.get('spike_reg', 0.001),
                    'log_dir': f'./logs/tsa/{dataset}/seed{seed}',
                    'checkpoint_dir': f'./checkpoints/tsa/{dataset}/seed{seed}',
                }

                Path(training_config['checkpoint_dir']).mkdir(
                    parents=True, exist_ok=True
                )
                Path(training_config['log_dir']).mkdir(
                    parents=True, exist_ok=True
                )

                # Train
                trainer = AdvancedTrainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    config=training_config,
                    device=device,
                    use_wandb=False,
                )

                metrics = trainer.train()

                all_results[dataset].append({
                    'seed': seed,
                    'test_acc': metrics['test_metrics']['acc'],
                    'test_energy': metrics['test_metrics'].get(
                        'avg_energy_uJ', 0.0
                    ),
                    'best_val_acc': metrics['best_val_acc'],
                })

                print(
                    f"  Seed {seed}: "
                    f"acc={metrics['test_metrics']['acc']:.4f}, "
                    f"energy={metrics['test_metrics'].get('avg_energy_uJ', 0):.4f}uJ"
                )

                # Save pretrained model
                pretrained_path = (
                    Path('./pretrained_models') /
                    f'tsa_{dataset}_seed{seed}.pth'
                )
                pretrained_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    model.state_dict(),
                    pretrained_path
                )
            except Exception as e:
                print(f"  ⚠️  Seed {seed} failed: {e}")
                all_results[dataset].append({'seed': seed, 'error': str(e)})

        # Print dataset summary
        accs = [
            r['test_acc'] for r in all_results[dataset]
            if 'test_acc' in r
        ]
        if accs:
            print(
                f"\n  {dataset.upper()} Summary: "
                f"{np.mean(accs)*100:.2f} ± "
                f"{np.std(accs)*100:.2f}%"
            )
        else:
            print(f"\n  {dataset.upper()} Summary: no successful runs")

    # Save results
    with open(save_dir / 'tsa_training_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    return all_results


# ─────────────────────────────────────────────
# Experiment 2: Ablations
# ─────────────────────────────────────────────

def run_ablations(
    config: dict,
    ablation_epochs: int,
    save_dir: Path,
    device: str,
) -> Dict:
    """
    Run all ablation studies on N-MNIST.
    Results go into paper Table 2.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: ABLATION STUDIES")
    print("=" * 60)

    # Use N-MNIST for ablations (faster)
    ds_config = config.get('nmnist', {})
    try:
        train_loader, val_loader, test_loader = get_loaders(
            'nmnist', ds_config
        )
    except Exception as e:
        print(f"  ⚠️  Could not load dataset 'nmnist': {e}")
        return {'error': str(e)}

    # Base config for ablations
    base_config = {
        'img_size': ds_config.get('img_size', 34),
        'patch_size': ds_config.get('patch_size', 2),
        'in_channels': ds_config.get('in_channels', 2),
        'num_classes': ds_config.get('num_classes', 10),
        'embed_dim': 256,
        'depth': 4,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'init_tau': 2.0,
    }

    # Override epochs for ablations
    ablation_config = config.copy()
    ablation_config['training'] = {
        **config.get('training', {}),
        'epochs': ablation_epochs,
    }

    ablation_framework = AblationFramework(
        base_config=base_config,
        dataset_loaders=(train_loader, val_loader, test_loader),
    )

    print("\nRunning 6 ablation studies...")
    print("(This takes time — each ablation trains multiple models)")

    ablation_results = ablation_framework.run_all_ablations()

    # Save
    with open(save_dir / 'ablation_results.json', 'w') as f:
        json.dump(ablation_results, f, indent=2, default=str)

    print(f"\nAblation results saved to {save_dir}/ablation_results.json")

    return ablation_results


# ─────────────────────────────────────────────
# Experiment 3: Hardware Validation
# ─────────────────────────────────────────────

def run_hardware_validation(
    datasets_list: List[str],
    config: dict,
    save_dir: Path,
    device: str,
) -> Dict:
    """
    Simulate Loihi 2 energy consumption.
    Results go into paper Figure 3.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: HARDWARE VALIDATION")
    print("(Loihi 2 Simulation)")
    print("=" * 60)

    simulator = LoihiSimulator()
    hardware_results = {}

    for dataset in datasets_list:
        print(f"\nDataset: {dataset.upper()}")

        ds_config = config.get(dataset, {})
        try:
            _, _, test_loader = get_loaders(dataset, ds_config)
        except Exception as e:
            print(f"  ⚠️  Could not load dataset '{dataset}': {e}")
            hardware_results[dataset] = {'error': str(e)}
            continue

        # Load best pretrained model if exists
        pretrained_path = (
            Path('./pretrained_models') /
            f'tsa_{dataset}_seed0.pth'
        )

        model = get_tsa_model(dataset, config)

        if pretrained_path.exists():
            ckpt = torch.load(pretrained_path, map_location=device)
            model.load_state_dict(ckpt)
            print(f"  Loaded: {pretrained_path}")
        else:
            print("  No pretrained model — using random weights")
            print("  Run training first for meaningful results")

        model = model.to(device)
        model.eval()

        # Simulate Loihi 2 execution
        results = simulator.simulate_execution(
            model=model,
            dataloader=test_loader,
            n_samples=100,
        )

        hardware_results[dataset] = results

        print(f"  Energy: {results['avg_energy_per_sample_uJ']:.4f} uJ")
        print(f"  Latency: {results['avg_latency_per_sample_ms']:.4f} ms")
        print(f"  Accuracy: {results['accuracy']:.4f}")

    # Save
    with open(save_dir / 'hardware_results.json', 'w') as f:
        json.dump(hardware_results, f, indent=2, default=str)

    print(f"\nHardware results saved to {save_dir}/hardware_results.json")

    return hardware_results


# ─────────────────────────────────────────────
# Experiment 4: Statistical Validation
# ─────────────────────────────────────────────

def run_statistical_validation(
    training_results: Dict,
    baseline_results_path: str,
    save_dir: Path,
) -> Dict:
    """
    Compare TSA vs baselines statistically.
    Paired t-test + Bonferroni correction.
    Results go into paper Table 1 footnotes.
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: STATISTICAL VALIDATION")
    print("=" * 60)

    validator = StatisticalValidator(
        results_dir=str(save_dir / 'statistical_analysis')
    )

    # Load baseline results if available
    baseline_path = Path(baseline_results_path)
    if not baseline_path.exists():
        print(
            "  ⚠️  Baseline results not found. "
            "Run run_baseline_comparison.py first."
        )
        return {}

    with open(baseline_path) as f:
        baseline_results = json.load(f)

    all_statistical_results = {}

    for dataset in training_results.keys():
        print(f"\nDataset: {dataset.upper()}")

        # Format TSA results (skip seeds that failed)
        tsa_results = [
            {
                'test_acc': r['test_acc'],
                'test_energy': r['test_energy'],
                'test_spikes': 0,
            }
            for r in training_results[dataset]
            if 'error' not in r
        ]

        if not tsa_results:
            print("  No successful TSA runs for this dataset")
            continue

        # Collect all method results
        all_methods = {'TSA_Ours': tsa_results}

        for algo, seed_results in baseline_results.get(
            dataset, {}
        ).items():
            if isinstance(seed_results, list):
                valid = [
                    {
                        'test_acc': r.get('test_acc', 0),
                        'test_energy': r.get('test_energy', 0),
                        'test_spikes': 0,
                    }
                    for r in seed_results
                    if 'error' not in r
                ]
                if valid:
                    all_methods[algo] = valid

        if len(all_methods) < 2:
            print("  Not enough methods for comparison")
            continue

        # Run statistical tests
        stat_results = validator.comprehensive_report(
            all_results=all_methods,
            save_dir=str(save_dir / 'statistical_analysis' / dataset)
        )

        all_statistical_results[dataset] = stat_results

    # Save
    with open(save_dir / 'statistical_results.json', 'w') as f:
        json.dump(
            all_statistical_results, f,
            indent=2, default=str
        )

    return all_statistical_results


# ─────────────────────────────────────────────
# Results Printer
# ─────────────────────────────────────────────

def print_tsa_summary(results: Dict):
    """Print TSA training summary."""
    print("\n" + "=" * 60)
    print("TSA RESULTS SUMMARY")
    print("=" * 60)

    for dataset, seed_results in results.items():
        if not isinstance(seed_results, list):
            continue

        valid = [r for r in seed_results if 'error' not in r]
        if not valid:
            print(f"\n{dataset.upper()}: no successful runs")
            continue

        accs = [r['test_acc'] for r in valid]
        energies = [r.get('test_energy', 0) for r in valid]

        print(f"\n{dataset.upper()}:")
        print(f"  Accuracy: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")
        print(f"  Energy:   {np.mean(energies):.4f} ± {np.std(energies):.4f} uJ")
        print(f"  Seeds:    {len(accs)}")

        for i, r in enumerate(seed_results):
            print(
                f"    Seed {i}: "
                f"acc={r['test_acc']*100:.2f}%, "
                f"energy={r.get('test_energy', 0):.4f}uJ"
            )


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Quick mode
    if args.quick:
        print("⚡ QUICK MODE: 1 seed, 5 epochs")
        args.n_seeds = 1
        args.epochs = 5
        args.ablation_epochs = 3
        args.datasets = ['nmnist']

    print("\n" + "=" * 60)
    print("TSA NEUROIPS — TSA EXPERIMENTS")
    print("=" * 60)
    print(f"Mode:      {args.mode}")
    print(f"Datasets:  {args.datasets}")
    print(f"Seeds:     {args.n_seeds}")
    print(f"Epochs:    {args.epochs}")
    print(f"Device:    {args.device}")

    # Setup
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    config = load_tsa_config()
    training_results = {}

    # Run selected experiments
    if args.mode in ['all', 'train']:
        training_results = run_training(
            datasets_list=args.datasets,
            n_seeds=args.n_seeds,
            epochs=args.epochs,
            config=config,
            save_dir=save_dir,
            device=args.device,
            load_checkpoint=args.load_checkpoint,
        )
        print_tsa_summary(training_results)

    if args.mode in ['all', 'ablations']:
        run_ablations(
            config=config,
            ablation_epochs=args.ablation_epochs,
            save_dir=save_dir,
            device=args.device,
        )

    if args.mode in ['all', 'hardware']:
        run_hardware_validation(
            datasets_list=args.datasets,
            config=config,
            save_dir=save_dir,
            device=args.device,
        )

    if args.mode in ['all', 'statistical']:
        # Load training results if not just computed
        if not training_results:
            results_path = save_dir / 'tsa_training_results.json'
            if results_path.exists():
                with open(results_path) as f:
                    training_results = json.load(f)

        run_statistical_validation(
            training_results=training_results,
            baseline_results_path=(
                './baseline_comparison_results/full_comparison.json'
            ),
            save_dir=save_dir,
        )

    print("\n✅ TSA experiments complete")
    print(f"   Results saved to {args.save_dir}/")


if __name__ == '__main__':
    main()