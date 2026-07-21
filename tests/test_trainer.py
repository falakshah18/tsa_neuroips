# tests/test_trainer.py
"""
Unit tests for trainer.
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tst_v2 import TemporalSpikingTransformer
from training.trainer_v2 import AdvancedTrainer


class TestTrainer:
    """Test training framework."""

    def test_advanced_trainer_creation(self):
        """Test trainer initialization."""
        model = TemporalSpikingTransformer(
            img_size=34,
            patch_size=2,
            in_channels=2,
            num_classes=10,
            embed_dim=32,
            depth=1,
            num_heads=2,
        )

        # Create dummy loaders
        train_loader = self._dummy_loader()
        val_loader = self._dummy_loader()
        test_loader = self._dummy_loader()

        config = {
            'epochs': 2,
            'lr': 0.001,
            'weight_decay': 0.05,
            'warmup_epochs': 1,
            'patience': 5,
            'mixed_precision': False,
            'gradient_accumulation_steps': 1,
            'spike_reg': 0.001,
            'log_dir': './test_logs',
            'checkpoint_dir': './test_checkpoints',
            'project_name': 'test',
            'run_name': 'test_run',
            'min_lr': 1e-6,
        }

        trainer = AdvancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            config=config,
            device='cpu',
            use_wandb=False,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_cpu_no_scaler(self):
        """Trainer on CPU must not create or use a GradScaler."""
        model = TemporalSpikingTransformer(
            img_size=34, patch_size=2, in_channels=2,
            num_classes=10, embed_dim=32, depth=1, num_heads=2,
        )
        config = {
            'epochs': 1, 'lr': 0.001, 'weight_decay': 0.05,
            'warmup_epochs': 0, 'patience': 1, 'mixed_precision': True,
            'gradient_accumulation_steps': 1, 'spike_reg': 0.0,
            'log_dir': './test_logs_cpu', 'checkpoint_dir': './test_ckpt_cpu',
            'project_name': 'test', 'run_name': 'test',
        }
        trainer = AdvancedTrainer(
            model=model,
            train_loader=self._dummy_loader(),
            val_loader=self._dummy_loader(),
            test_loader=self._dummy_loader(),
            config=config,
            device='cpu',
            use_wandb=False,
        )
        # AMP should be disabled on CPU regardless of config
        assert trainer.use_amp is False
        # No scaler attribute after the scaler was removed
        assert not hasattr(trainer, 'scaler')

    def test_checkpoint_save_load(self):
        """Saved checkpoint can be loaded back, restoring model weights."""
        model = TemporalSpikingTransformer(
            img_size=34, patch_size=2, in_channels=2,
            num_classes=10, embed_dim=32, depth=1, num_heads=2,
        )
        config = {
            'epochs': 1, 'lr': 0.001, 'weight_decay': 0.05,
            'warmup_epochs': 0, 'patience': 1, 'mixed_precision': False,
            'gradient_accumulation_steps': 1, 'spike_reg': 0.0,
            'log_dir': './test_logs_ckpt', 'checkpoint_dir': './test_ckpt_ckpt',
            'project_name': 'test', 'run_name': 'test',
        }
        trainer = AdvancedTrainer(
            model=model,
            train_loader=self._dummy_loader(),
            val_loader=self._dummy_loader(),
            test_loader=self._dummy_loader(),
            config=config,
            device='cpu',
            use_wandb=False,
        )
        # Save
        trainer.save_checkpoint(epoch=0, is_best=True)
        best_path = Path(config['checkpoint_dir']) / 'best.pth'
        assert best_path.exists()

        # Load into a fresh trainer and verify weights match
        model2 = TemporalSpikingTransformer(
            img_size=34, patch_size=2, in_channels=2,
            num_classes=10, embed_dim=32, depth=1, num_heads=2,
        )
        trainer2 = AdvancedTrainer(
            model=model2,
            train_loader=self._dummy_loader(),
            val_loader=self._dummy_loader(),
            test_loader=self._dummy_loader(),
            config=config,
            device='cpu',
            use_wandb=False,
        )
        epoch = trainer2.load_checkpoint(str(best_path))
        assert epoch == 0
        for p1, p2 in zip(model.parameters(), model2.parameters()):
            assert torch.equal(p1, p2), "Loaded weights differ from saved"

    def test_count_spikes_robustness(self):
        """_count_spikes must handle arbitrary dicts gracefully."""
        model = TemporalSpikingTransformer(
            img_size=34, patch_size=2, in_channels=2,
            num_classes=10, embed_dim=32, depth=1, num_heads=2,
        )
        trainer = AdvancedTrainer(
            model=model,
            train_loader=self._dummy_loader(),
            val_loader=self._dummy_loader(),
            test_loader=self._dummy_loader(),
            config={'epochs': 1, 'lr': 0.001, 'log_dir': './t', 'checkpoint_dir': './t'},
            device='cpu',
            use_wandb=False,
        )
        assert trainer._count_spikes({}) == 0.0
        assert trainer._count_spikes({'total_spikes': 42.0}) == 42.0
        assert trainer._count_spikes('not a dict') == 0.0
        assert trainer._count_spikes({'blocks': [None, 7]}) == 0.0

    def test_dummy_loader(self):
        """Verify dummy loader yields correct shapes."""
        loader = self._dummy_loader()
        x, y = next(iter(loader))
        assert x.dim() == 5  # [B, T, C, H, W]
        assert y.dim() == 1  # [B]

    def _dummy_loader(self):
        """Create dummy data loader with correct [T, B, C, H, W] shape."""
        from torch.utils.data import DataLoader, TensorDataset

        T, B = 5, 2
        # Stack T frames per sample so each dataset item is [T, C, H, W]
        x = torch.randn(B * 4, T, 2, 34, 34)   # [N_samples, T, C, H, W]
        y = torch.randint(0, 10, (B * 4,))

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=B, shuffle=False)