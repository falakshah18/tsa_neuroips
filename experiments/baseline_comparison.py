# experiments/baseline_comparison.py
"""
Core baseline comparison framework.
Evaluates all algorithm families on multiple datasets.
"""

import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import yaml

from baselines.surrogate_gradient import get_surrogate_grad_model
from baselines.ann_to_snn import get_ann_to_snn_model, ANNtoSNNConverter, ConvertedSNN
from baselines.stdp import get_stdp_model
from baselines.eprop import get_eprop_model
from baselines.temporal_coding import get_ttfs_model
from models.tst_v2 import TemporalSpikingTransformer

# Dataset imports
from tonic import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from spikingjelly.activation_based import functional


class _ANNTimeAverageWrapper(torch.nn.Module):
    """
    Adapts a plain (non-spiking) SourceANN/SourceANN_SHD for training through
    AdvancedTrainer, which always feeds [T, B, ...] and expects a
    (logits, metrics) tuple back.

    SourceANN itself expects a single averaged frame [B, C, H, W] (see its
    docstring: "for neuromorphic data, average over T first") and returns a
    bare tensor. This wrapper does the averaging and tuple-wrapping so the
    same generic trainer can be reused, while leaving the wrapped `ann`'s
    parameters/identity untouched (training this wrapper trains `ann` itself,
    since it's held by reference, not copied).
    """

    def __init__(self, ann: torch.nn.Module):
        super().__init__()
        self.ann = ann

    def forward(self, x: torch.Tensor):
        if x.dim() >= 3:
            x = x.mean(dim=0)  # [T, B, ...] -> [B, ...]
        logits = self.ann(x)
        metrics = {'total_spikes': 0.0, 'blocks': []}  # plain ANN: no spikes
        return logits, metrics


class BaselineComparison:
    """
    Systematic comparison of all neuromorphic algorithm families.
    This is the main UGRP contribution.
    """

    # Canonical internal algorithm identifiers, and a lookup table that
    # normalizes both the CLI's display-style names (e.g. 'TTFS',
    # 'Supervised_STDP') and casual lowercase aliases (e.g. 'stdp',
    # 'surrogate_grad') down to them. Without this, --algorithm/--algorithms
    # was silently ignored: run() always iterated a hardcoded list of all 6,
    # regardless of what was requested.
    ALL_ALGORITHMS = ['surrogate_gradient', 'ann_to_snn', 'stdp', 'eprop', 'ttfs', 'tsa']
    _ALGO_ALIASES = {
        'surrogate_gradient': 'surrogate_gradient', 'surrogate_grad': 'surrogate_gradient',
        'Surrogate_Gradient': 'surrogate_gradient',
        'ann_to_snn': 'ann_to_snn', 'ANN_to_SNN': 'ann_to_snn',
        'stdp': 'stdp', 'supervised_stdp': 'stdp', 'Supervised_STDP': 'stdp',
        'eprop': 'eprop', 'e_prop': 'eprop', 'E_prop': 'eprop',
        'ttfs': 'ttfs', 'TTFS': 'ttfs',
        'tsa': 'tsa', 'tsa_ours': 'tsa', 'TSA_Ours': 'tsa',
    }

    @classmethod
    def _normalize_algorithms(cls, algorithms):
        if not algorithms:
            return list(cls.ALL_ALGORITHMS)
        normalized = []
        for a in algorithms:
            canonical = cls._ALGO_ALIASES.get(a)
            if canonical is None:
                raise ValueError(
                    f"Unknown algorithm '{a}'. Valid options: "
                    f"{sorted(set(cls._ALGO_ALIASES.keys()))}"
                )
            if canonical not in normalized:
                normalized.append(canonical)
        return normalized

    def __init__(
        self,
        datasets: List[str] = ['nmnist', 'shd'],
        algorithms: List[str] = None,
        n_seeds: int = 3,
        epochs: int = 100,
        save_dir: str = './baseline_comparison_results',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        quick: bool = False,
    ):
        self.datasets = datasets
        self.algorithms = self._normalize_algorithms(algorithms)
        self.n_seeds = n_seeds
        self.epochs = epochs
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.quick = quick
        self.results = {}
        self.bio_scores = {}

        # Load configs
        self.configs = self._load_configs()

    @staticmethod
    def _subsample(dataset, n: int):
        """
        Truncate a dataset to at most n samples, for quick-mode smoke tests.

        Without this, --quick only reduces epochs/seeds but still iterates
        the *entire* real dataset every epoch (e.g. all ~60k N-MNIST
        training images), which can take hours on CPU -- defeating the
        purpose of a fast sanity check. Uses a fixed (unshuffled) prefix
        rather than a random sample, so quick runs are also deterministic
        and reproducible run-to-run.
        """
        n = min(n, len(dataset))
        return Subset(dataset, list(range(n)))

    def _load_configs(self) -> dict:
        """Load algorithm configs."""
        config_dir = Path(__file__).parent.parent / 'configs'
        configs = {}

        for algo in ['surrogate_grad', 'ann_to_snn', 'stdp', 'eprop', 'ttfs', 'tsa']:
            config_path = config_dir / f'{algo}_config.yaml'
            if config_path.exists():
                with open(config_path) as f:
                    configs[algo] = yaml.safe_load(f)
            else:
                configs[algo] = {}

        return configs

    def _get_data_loaders(self, dataset: str, config: dict) -> Tuple:
        """Get data loaders for a dataset."""
        if dataset == 'nmnist':
            sensor_size = (34, 34, 2)
            n_time_bins = config.get('T', 20)
            transform = transforms.Compose([
                transforms.Denoise(filter_time=10000),
                transforms.ToFrame(
                    sensor_size=sensor_size,
                    n_time_bins=n_time_bins
                ),
            ])

            train_ds = datasets.NMNIST(
                save_to='./data', train=True, transform=transform
            )
            test_ds = datasets.NMNIST(
                save_to='./data', train=False, transform=transform
            )

            train_size = int(0.9 * len(train_ds))
            val_size = len(train_ds) - train_size
            train_ds, val_ds = random_split(train_ds, [train_size, val_size])

            if self.quick:
                train_ds = self._subsample(train_ds, 320)
                val_ds = self._subsample(val_ds, 128)
                test_ds = self._subsample(test_ds, 128)

            batch_size = config.get('batch_size', 32)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

            return train_loader, val_loader, test_loader

        elif dataset == 'shd':
            # tonic's actual SHD sensor_size has no polarity channel (P=1),
            # unlike DVS-style vision sensors. See tonic.datasets.SHD.sensor_size.
            sensor_size = (700, 1, 1)
            n_time_bins = config.get('T', 100)
            transform = transforms.Compose([
                transforms.ToFrame(
                    sensor_size=sensor_size,
                    n_time_bins=n_time_bins
                ),
                # ToFrame yields [T, P, X, Y] = [T, 1, 700, 1]; the SHD model
                # variants expect a flat [T, input_size] per sample.
                lambda frame: frame.reshape(frame.shape[0], -1),
            ])

            train_ds = datasets.SHD(
                save_to='./data', train=True, transform=transform
            )
            test_ds = datasets.SHD(
                save_to='./data', train=False, transform=transform
            )

            train_size = int(0.9 * len(train_ds))
            val_size = len(train_ds) - train_size
            train_ds, val_ds = random_split(train_ds, [train_size, val_size])

            if self.quick:
                train_ds = self._subsample(train_ds, 320)
                val_ds = self._subsample(val_ds, 128)
                test_ds = self._subsample(test_ds, 128)

            batch_size = config.get('batch_size', 32)

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=4, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )

            return train_loader, val_loader, test_loader

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    def _create_model(
        self,
        algorithm: str,
        dataset: str,
        config: dict,
    ) -> torch.nn.Module:
        """Create model for a specific algorithm and dataset."""
        if algorithm == 'surrogate_gradient':
            return get_surrogate_grad_model(dataset, config)

        elif algorithm == 'ann_to_snn':
            ann, snn_class, snn_kwargs = get_ann_to_snn_model(dataset, config)
            # Return the ANN for training
            return ann

        elif algorithm == 'stdp':
            return get_stdp_model(dataset, config)

        elif algorithm == 'eprop':
            return get_eprop_model(dataset, config)

        elif algorithm == 'ttfs':
            return get_ttfs_model(dataset, config)

        elif algorithm == 'tsa':
            ds_config = config.get(dataset, {})
            return TemporalSpikingTransformer(
                img_size=ds_config.get('img_size', 34),
                patch_size=ds_config.get('patch_size', 2),
                in_channels=ds_config.get('in_channels', 2),
                num_classes=ds_config.get('num_classes', 10),
                embed_dim=config.get('model', {}).get('embed_dim', 256),
                depth=config.get('model', {}).get('depth', 4),
                num_heads=config.get('model', {}).get('num_heads', 8),
                mlp_ratio=config.get('model', {}).get('mlp_ratio', 4.0),
                init_tau=config.get('model', {}).get('init_tau', 2.0),
            )

        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def _train_ann_to_snn(
        self,
        ann: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: dict,
        seed: int,
        dataset: str,
    ) -> Dict:
        """Special training for ANN-to-SNN: train ANN, then convert."""
        from training.trainer_v2 import AdvancedTrainer

        # Train ANN
        ann_config = {
            'epochs': config.get('ann', {}).get('epochs', 50),
            'lr': config.get('ann', {}).get('lr', 0.001),
            'weight_decay': config.get('ann', {}).get('weight_decay', 0.0001),
            'warmup_epochs': 5,
            'patience': 15,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
            'label_smoothing': 0.0,
            'grad_clip': 1.0,
            'spike_reg': 0.0,
            'log_dir': f'./logs/ann_to_snn/seed{seed}',
            'checkpoint_dir': f'./checkpoints/ann_to_snn/seed{seed}',
            'project_name': 'ANN_to_SNN',
            'run_name': f'ANN_to_SNN_seed{seed}',
        }

        trainer = AdvancedTrainer(
            model=_ANNTimeAverageWrapper(ann),
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=ann_config,
            device=self.device,
            use_wandb=False,
        )

        train_result = trainer.train()

        if dataset == 'shd':
            # get_ann_to_snn_model() returns snn_class=None for SHD -- there's
            # no FC-based ConvertedSNN equivalent, only the vision (Conv2d)
            # ConvertedSNN, which is architecturally incompatible with
            # SourceANN_SHD's flat FC layers. Evaluate the trained ANN
            # directly instead of attempting a conversion.
            eval_model = _ANNTimeAverageWrapper(ann).to(self.device)
            eval_model.eval()

            total_acc = 0
            total_samples = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(self.device).float()
                    target = target.to(self.device)
                    if data.dim() >= 3:
                        data = data.transpose(0, 1).contiguous()  # [B,T,...] -> [T,B,...]

                    logits, _ = eval_model(data)
                    pred = logits.argmax(dim=1)
                    acc = (pred == target).float().mean()

                    batch_size = target.shape[0]
                    total_acc += acc.item() * batch_size
                    total_samples += batch_size

            return {
                'test_acc': total_acc / total_samples,
                'test_energy': 0.0,  # plain ANN, no spikes to count
                'total_spikes': 0.0,
                'train_time': train_result.get('train_time', 0),
                'best_val_acc': train_result.get('best_val_acc', 0),
            }

        # Convert to SNN (vision datasets only)
        converter = ANNtoSNNConverter(
            percentile=config.get('conversion', {}).get('percentile', 99.9)
        )

        snn_kwargs = {
            'in_channels': config.get('in_channels', 2),
            'num_classes': config.get('num_classes', 10),
            'img_size': config.get('img_size', 34),
            'T': config.get('conversion', {}).get('T_inference', 100),
        }

        snn = converter.convert(
            ann=ann,
            snn_class=ConvertedSNN,
            snn_kwargs=snn_kwargs,
            dataloader=train_loader,
            device=self.device,
        )

        # Evaluate SNN
        snn = snn.to(self.device)
        snn.eval()

        total_acc = 0
        total_samples = 0
        total_spikes = 0

        with torch.no_grad():
            for data, target in test_loader:
                data = data.to(self.device).float()
                target = target.to(self.device)

                if data.dim() == 5:
                    # DataLoader yields [B, T, C, H, W]; ConvertedSNN expects [T, B, C, H, W].
                    data = data.transpose(0, 1).contiguous()

                logits, metrics = snn(data)
                pred = logits.argmax(dim=1)
                acc = (pred == target).float().mean()

                batch_size = target.shape[0]
                total_acc += acc.item() * batch_size
                total_samples += batch_size
                total_spikes += metrics['total_spikes']

                functional.reset_net(snn)

        return {
            'test_acc': total_acc / total_samples,
            'test_energy': (total_spikes / total_samples) * 0.1e-12,
            'total_spikes': total_spikes,
            'train_time': train_result.get('train_time', 0),
            'best_val_acc': train_result.get('best_val_acc', 0),
        }

    def _run_seed(
        self,
        algorithm: str,
        dataset: str,
        seed: int,
    ) -> Dict:
        """Run a single seed experiment."""
        print(f"  Running {algorithm} on {dataset}, seed {seed}")

        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Get config
        algo_config = self.configs.get(algorithm, {})
        ds_config = algo_config.get(dataset, {})

        # Get data loaders
        train_loader, val_loader, test_loader = self._get_data_loaders(
            dataset, ds_config
        )

        # Special handling for ANN-to-SNN
        if algorithm == 'ann_to_snn':
            ann = self._create_model(algorithm, dataset, ds_config)
            ann = ann.to(self.device)
            return self._train_ann_to_snn(
                ann, train_loader, val_loader, test_loader,
                algo_config, seed, dataset
            )

        # Create model
        model = self._create_model(algorithm, dataset, ds_config)
        model = model.to(self.device)

        # Training config
        train_config = algo_config.get('training', {})
        training_config = {
            'epochs': self.epochs,
            'lr': train_config.get('lr', 0.001),
            'weight_decay': train_config.get('weight_decay', 0.05),
            'warmup_epochs': train_config.get('warmup_epochs', 10),
            'patience': train_config.get('patience', 20),
            'mixed_precision': train_config.get('mixed_precision', True),
            'gradient_accumulation_steps': train_config.get(
                'gradient_accumulation_steps', 1
            ),
            'spike_reg': train_config.get('spike_reg', 0.0),
            'label_smoothing': train_config.get('label_smoothing', 0.0),
            'grad_clip': train_config.get('grad_clip', 1.0),
            'log_dir': f'./logs/{algorithm}/{dataset}/seed{seed}',
            'checkpoint_dir': f'./checkpoints/{algorithm}/{dataset}/seed{seed}',
            'project_name': f'{algorithm.upper()}_NEUROIPS',
            'run_name': f'{algorithm}_{dataset}_seed{seed}',
        }

        from training.trainer_v2 import AdvancedTrainer

        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=training_config,
            device=self.device,
            use_wandb=False,
        )

        start_time = time.time()
        result = trainer.train()
        train_time = time.time() - start_time

        return {
            'test_acc': result['test_metrics']['acc'],
            'test_energy': result['test_metrics'].get('avg_energy_uJ', 0),
            'train_time': train_time / 60,  # minutes
            'best_val_acc': result['best_val_acc'],
        }

    def _score_bio_plausibility(self, algorithm: str) -> Dict:
        """Score biological plausibility using 5-criterion rubric."""
        # Load bio scores from config
        algo_config = self.configs.get(algorithm, {})
        bio = algo_config.get('bio_plausibility', {})

        if not bio:
            # Default scores
            bio_scores = {
                'surrogate_gradient': {
                    'local_learning_rule': False,
                    'spike_based_communication': True,
                    'temporal_dynamics': True,
                    'no_weight_transport': False,
                    'online_learning': False,
                },
                'ann_to_snn': {
                    'local_learning_rule': False,
                    'spike_based_communication': True,
                    'temporal_dynamics': True,
                    'no_weight_transport': False,
                    'online_learning': False,
                },
                'stdp': {
                    'local_learning_rule': True,
                    'spike_based_communication': True,
                    'temporal_dynamics': True,
                    'no_weight_transport': True,
                    'online_learning': False,
                },
                'eprop': {
                    'local_learning_rule': True,
                    'spike_based_communication': True,
                    'temporal_dynamics': True,
                    'no_weight_transport': True,
                    'online_learning': False,
                },
                'ttfs': {
                    'local_learning_rule': False,
                    'spike_based_communication': True,
                    'temporal_dynamics': True,
                    'no_weight_transport': False,
                    'online_learning': False,
                },
                'tsa': {
                    'local_learning_rule': False,
                    'spike_based_communication': True,
                    'temporal_dynamics': True,
                    'no_weight_transport': False,
                    'online_learning': False,
                },
            }
            bio = bio_scores.get(algorithm, {})

        # Config YAMLs may already embed a precomputed 'total_score' alongside
        # the boolean criteria; exclude it so we don't double-count it as a
        # truthy criterion when summing.
        criteria = {k: v for k, v in bio.items() if k != 'total_score'}

        total = sum(1 for v in criteria.values() if v)
        result = {
            'criteria': criteria,
            'total_score': total,
        }
        # Keep flat boolean keys too for any callers that expect them directly.
        result.update(bio)

        return result

    def run(self) -> Dict:
        """Run the full baseline comparison."""
        print("\n" + "=" * 70)
        print("RUNNING BASELINE COMPARISON")
        print("=" * 70)

        algorithms = self.algorithms

        for dataset in self.datasets:
            print(f"\n{'─'*70}")
            print(f"Dataset: {dataset.upper()}")
            print(f"{'─'*70}")

            self.results[dataset] = {}

            for algorithm in algorithms:
                print(f"\nAlgorithm: {algorithm}")

                algo_results = []
                for seed in range(self.n_seeds):
                    try:
                        result = self._run_seed(algorithm, dataset, seed)
                        result['seed'] = seed
                        result['error'] = False
                        algo_results.append(result)
                    except Exception as e:
                        print(f"  ⚠️  Seed {seed} failed: {e}")
                        algo_results.append({'seed': seed, 'error': str(e)})

                self.results[dataset][algorithm] = algo_results

                # Print summary
                valid = [r for r in algo_results if not r.get('error', False)]
                if valid:
                    accs = [r['test_acc'] for r in valid]
                    energies = [r.get('test_energy', 0) for r in valid]
                    print(f"  Acc: {np.mean(accs)*100:.2f} ± {np.std(accs)*100:.2f}%")
                    print(f"  Energy: {np.mean(energies):.4f} ± {np.std(energies):.4f} μJ")

        # Compute bio scores
        for algorithm in algorithms:
            self.bio_scores[algorithm] = self._score_bio_plausibility(algorithm)
        self.results['bio_plausibility'] = self.bio_scores

        # Save results
        self._save_results()

        return self.results

    def _save_results(self):
        """Save results to JSON files."""
        for dataset in self.datasets:
            if dataset in self.results:
                path = self.save_dir / f'{dataset}_results.json'
                with open(path, 'w') as f:
                    json.dump(self.results[dataset], f, indent=2)

        # Full results
        with open(self.save_dir / 'full_comparison.json', 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✅ Results saved to {self.save_dir}/")

    def generate_report(self) -> Dict:
        """Generate aggregated report."""
        report = {}

        for dataset in self.datasets:
            if dataset not in self.results:
                continue

            report[dataset] = {}
            for algorithm, results in self.results[dataset].items():
                if algorithm == 'bio_plausibility':
                    continue

                valid = [r for r in results if not r.get('error', False)]
                if valid:
                    report[dataset][algorithm] = {
                        'mean_acc': np.mean([r['test_acc'] for r in valid]),
                        'std_acc': np.std([r['test_acc'] for r in valid]),
                        'mean_energy': np.mean([r.get('test_energy', 0) for r in valid]),
                        'std_energy': np.std([r.get('test_energy', 0) for r in valid]),
                        'mean_train_time': np.mean([r.get('train_time', 0) for r in valid]),
                        'n_seeds': len(valid),
                        'best_val_acc': max([r.get('best_val_acc', 0) for r in valid]),
                    }

        report['bio_plausibility'] = self.bio_scores

        # Save report
        with open(self.save_dir / 'ugrp_report.json', 'w') as f:
            json.dump(report, f, indent=2)

        return report