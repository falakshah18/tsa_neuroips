# tests/test_trainer.py
"""
Unit tests for trainer.
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

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

    def _dummy_loader(self):
        """Create dummy data loader with correct [T, B, C, H, W] shape."""
        from torch.utils.data import DataLoader, TensorDataset

        T, B = 5, 2
        # Stack T frames per sample so each dataset item is [T, C, H, W]
        x = torch.randn(B * 4, T, 2, 34, 34)   # [N_samples, T, C, H, W]
        y = torch.randint(0, 10, (B * 4,))

        dataset = TensorDataset(x, y)
        return DataLoader(dataset, batch_size=B, shuffle=False)