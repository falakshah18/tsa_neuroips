# tests/test_tsa.py
"""
Unit tests for TSA model.
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.tst_v2 import (
    LearnableTSA,
    TSABlock,
    TemporalSpikingTransformer,
)


class TestTSA:
    """Test TSA components."""

    def test_learnable_tsa(self):
        """Test Learnable TSA module."""
        dim = 64
        num_heads = 4
        T = 10
        B = 2
        N = 16  # number of patches

        tsa = LearnableTSA(dim=dim, num_heads=num_heads)
        x = torch.randn(T, B, N, dim)

        output, metrics = tsa(x)
        assert output.shape == (T, B, N, dim)
        assert 'attention' in metrics
        assert 'total_spikes' in metrics['attention']

    def test_tsa_block(self):
        """Test TSA block."""
        dim = 64
        block = TSABlock(dim=dim, num_heads=4)
        T, B, N = 5, 2, 16
        x = torch.randn(T, B, N, dim)

        output, metrics = block(x)
        assert output.shape == (T, B, N, dim)
        assert 'attention' in metrics

    def test_temporal_spiking_transformer(self):
        """Test full TST model."""
        model = TemporalSpikingTransformer(
            img_size=34,
            patch_size=2,
            in_channels=2,
            num_classes=10,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )

        T, B = 5, 2
        x = torch.randn(T, B, 2, 34, 34)

        output, metrics = model(x)
        assert output.shape == (B, 10)

        # Test energy breakdown
        energy = model.get_energy_breakdown(x)
        assert 'total_energy_J' in energy
        assert 'energy_per_sample_uJ' in energy

    def test_learnable_parameters(self):
        """Test that TSA parameters are learnable."""
        model = TemporalSpikingTransformer(
            img_size=34,
            patch_size=2,
            in_channels=2,
            num_classes=10,
            embed_dim=64,
            depth=2,
            num_heads=4,
        )

        # Check that neuron parameters are learnable
        for name, param in model.named_parameters():
            if 'tau' in name or 'threshold' in name or 'temperature' in name:
                assert param.requires_grad

        # Check that pos_embed is learnable
        assert model.pos_embed.requires_grad