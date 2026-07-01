# tests/test_baselines.py
"""
Unit tests for baseline implementations.
"""

import torch
import pytest
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from baselines.surrogate_gradient import get_surrogate_grad_model
from baselines.stdp import get_stdp_model
from baselines.eprop import get_eprop_model
from baselines.temporal_coding import get_ttfs_model
from baselines.ann_to_snn import get_ann_to_snn_model, ANNtoSNNConverter
from spikingjelly.activation_based import functional


class TestBaselines:
    """Test baseline models."""

    @pytest.mark.parametrize('dataset', ['nmnist', 'shd'])
    def test_surrogate_gradient(self, dataset):
        """Test surrogate gradient model."""
        config = {'T': 5, 'batch_size': 2}
        if dataset == 'nmnist':
            config.update({
                'in_channels': 2,
                'num_classes': 10,
                'img_size': 34,
            })
            model = get_surrogate_grad_model(dataset, config)
            x = torch.randn(5, 2, 2, 34, 34)  # [T, B, C, H, W]
        else:
            config.update({
                'input_size': 700,
                'num_classes': 20,
            })
            model = get_surrogate_grad_model(dataset, config)
            x = torch.randn(5, 2, 700)  # [T, B, input_size]

        output, metrics = model(x)
        functional.reset_net(model)
        assert output.shape[1] == config['num_classes']
        assert 'total_spikes' in metrics

    @pytest.mark.parametrize('dataset', ['nmnist', 'shd'])
    def test_stdp(self, dataset):
        """Test STDP model."""
        config = {'T': 5, 'batch_size': 2}
        if dataset == 'nmnist':
            config.update({
                'in_channels': 2,
                'num_classes': 10,
                'img_size': 34,
            })
            model = get_stdp_model(dataset, config)
            x = torch.randn(5, 2, 2, 34, 34)
        else:
            config.update({
                'input_size': 700,
                'num_classes': 20,
            })
            model = get_stdp_model(dataset, config)
            x = torch.randn(5, 2, 700)

        output, metrics = model(x)
        functional.reset_net(model)
        assert output.shape[1] == config['num_classes']

    @pytest.mark.parametrize('dataset', ['nmnist', 'shd'])
    def test_eprop(self, dataset):
        """Test E-prop model."""
        config = {'T': 5, 'batch_size': 2}
        if dataset == 'nmnist':
            config.update({
                'in_channels': 2,
                'num_classes': 10,
                'img_size': 34,
            })
            model = get_eprop_model(dataset, config)
            x = torch.randn(5, 2, 2, 34, 34)
        else:
            config.update({
                'input_size': 700,
                'num_classes': 20,
            })
            model = get_eprop_model(dataset, config)
            x = torch.randn(5, 2, 700)

        output, metrics = model(x)
        functional.reset_net(model)
        assert output.shape[1] == config['num_classes']

    @pytest.mark.parametrize('dataset', ['nmnist', 'shd'])
    def test_ttfs(self, dataset):
        """Test TTFS model."""
        config = {'T': 5, 'batch_size': 2}
        if dataset == 'nmnist':
            config.update({
                'in_channels': 2,
                'num_classes': 10,
                'img_size': 34,
            })
            model = get_ttfs_model(dataset, config)
            x = torch.randn(5, 2, 2, 34, 34)
        else:
            config.update({
                'input_size': 700,
                'num_classes': 20,
            })
            model = get_ttfs_model(dataset, config)
            x = torch.randn(5, 2, 700)

        output, metrics = model(x)
        functional.reset_net(model)
        assert output.shape[1] == config['num_classes']
    def test_ann_to_snn(self):
        """Test ANN-to-SNN conversion pipeline (nmnist only — vision path)."""
        from baselines.ann_to_snn import get_ann_to_snn_model, ANNtoSNNConverter

        config = {
            'T': 10, 'batch_size': 2,
            'in_channels': 2, 'num_classes': 10, 'img_size': 34,
        }
        ann, snn_class, snn_kwargs = get_ann_to_snn_model('nmnist', config, device='cpu')

        # Train ANN for 1 step to get non-random weights
        x_ann = torch.randn(2, 2, 34, 34)
        out_ann = ann(x_ann)
        assert out_ann.shape == (2, 10)

        # Build SNN with default thresholds (no calibration data needed for shape test)
        snn = snn_class(**snn_kwargs)
        functional.set_step_mode(snn, 'm')

        x_snn = torch.randn(10, 2, 2, 34, 34)
        out_snn, metrics = snn(x_snn)
        assert out_snn.shape == (2, 10)
        assert 'total_spikes' in metrics

    @pytest.mark.parametrize('dataset', ['nmnist', 'shd'])
    def test_eprop_training_mode(self, dataset):
        """Test E-prop model with training=True (eprop update path)."""
        config = {'T': 5, 'batch_size': 2}
        if dataset == 'nmnist':
            config.update({'in_channels': 2, 'num_classes': 10, 'img_size': 34})
            model = get_eprop_model(dataset, config)
            x = torch.randn(5, 2, 2, 34, 34)
            targets = torch.randint(0, 10, (2,))
        else:
            config.update({'input_size': 700, 'num_classes': 20})
            model = get_eprop_model(dataset, config)
            x = torch.randn(5, 2, 700)
            targets = torch.randint(0, 20, (2,))

        output, metrics = model(x, targets=targets, training=True)
        functional.reset_net(model)
        assert output.shape[1] == config['num_classes']

    @pytest.mark.parametrize('dataset', ['nmnist', 'shd'])
    def test_stdp_training_mode(self, dataset):
        """Test Supervised STDP model with training=True (stdp update path)."""
        config = {'T': 5, 'batch_size': 2}
        if dataset == 'nmnist':
            config.update({'in_channels': 2, 'num_classes': 10, 'img_size': 34})
            model = get_stdp_model(dataset, config)
            x = torch.randn(5, 2, 2, 34, 34)
            targets = torch.randint(0, 10, (2,))
        else:
            config.update({'input_size': 700, 'num_classes': 20})
            model = get_stdp_model(dataset, config)
            x = torch.randn(5, 2, 700)
            targets = torch.randint(0, 20, (2,))

        output, metrics = model(x, targets=targets, training=True)
        functional.reset_net(model)
        assert output.shape[1] == config['num_classes']   