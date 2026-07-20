# baselines/surrogate_gradient/surrogate_grad_snn.py
"""
Baseline 1: Surrogate Gradient SNN
Most common modern SNN training method.
Uses ATan surrogate for backprop through spikes.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple, Dict


class SurrogateGradSNN(nn.Module):
    """
    Standard SNN trained with surrogate gradients.
    Architecture: Conv -> LIF -> Conv -> LIF -> FC -> LIF -> FC
    
    Reference:
        Neftci et al. (2019) "Surrogate Gradient Learning in SNNs"
    
    Used as baseline against TSA.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        img_size: int = 34,
        tau: float = 2.0,
        T: int = 20,
    ):
        super().__init__()

        self.T = T
        self.num_classes = num_classes

        # Surrogate function
        surrogate_fn = surrogate.ATan()

        # Feature extractor
        self.features = nn.Sequential(
            # Block 1
            layer.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn, detach_reset=True),
            layer.MaxPool2d(2, 2),

            # Block 2
            layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn, detach_reset=True),
            layer.MaxPool2d(2, 2),

            # Block 3
            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn, detach_reset=True),
        )

        # Compute feature size
        feature_size = (img_size // 4) ** 2 * 128

        # Classifier
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(feature_size, 512),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn, detach_reset=True),
            layer.Dropout(0.5),
            layer.Linear(512, num_classes),
        )

        # Multi-step mode
        functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [T, B, C, H, W] spike input
        Returns:
            logits: [B, num_classes]
            metrics: spike counts, energy estimate
        """
        # Feature extraction
        x = self.features(x)   # [T, B, 128, H', W']

        # Classification
        x = self.classifier(x)  # [T, B, num_classes]

        # Average over time
        out = x.mean(0)  # [B, num_classes]

        # In multi-step mode, m.spike is [T, B, N] — sum correctly
        total_spikes = 0
        for m in self.modules():
            if isinstance(m, neuron.LIFNode):
                if hasattr(m, 'spike') and m.spike is not None:
                    total_spikes += m.spike.sum().item()

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (self.T * out.shape[0]),
            'avg_energy_uJ': total_spikes * 0.1e-6,  # Loihi 2: 0.1pJ per spike → μJ
        }

        return out, metrics


class SurrogateGradSNN_SHD(nn.Module):
    """
    Surrogate Gradient SNN for SHD (audio) dataset.
    1D temporal data, different architecture.
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_size: int = 256,
        num_classes: int = 20,
        tau: float = 2.0,
        T: int = 100,
        num_layers: int = 3,
    ):
        super().__init__()

        self.T = T
        surrogate_fn = surrogate.ATan()

        layers = []

        # Input layer
        layers += [
            layer.Linear(input_size, hidden_size),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn, detach_reset=True),
            layer.Dropout(0.3),
        ]

        # Hidden layers
        for _ in range(num_layers - 1):
            layers += [
                layer.Linear(hidden_size, hidden_size),
                neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn, detach_reset=True),
                layer.Dropout(0.3),
            ]

        self.network = nn.Sequential(*layers)

        # Output
        self.head = layer.Linear(hidden_size, num_classes)

        functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [T, B, input_size]
        Returns:
            logits: [B, num_classes]
            metrics: dict
        """
        x = self.network(x)     # [T, B, hidden]
        x = self.head(x)        # [T, B, num_classes]

        total_spikes = sum(
            m.spike.sum().item()
            for m in self.modules()
            if isinstance(m, neuron.LIFNode) and hasattr(m, 'spike') and m.spike is not None
        )

        out = x.mean(0)

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (self.T * x.shape[1]),
            'avg_energy_uJ': total_spikes * 0.1e-6,
        }

        return out, metrics


def get_surrogate_grad_model(dataset: str, config: dict) -> nn.Module:
    """
    Factory function — returns correct model for dataset.
    """
    if dataset == 'shd':
        return SurrogateGradSNN_SHD(
            input_size=config.get('input_size', 700),
            hidden_size=config.get('hidden_size', 256),
            num_classes=config.get('num_classes', 20),
            tau=config.get('tau', 2.0),
            T=config.get('T', 100),
        )
    else:
        return SurrogateGradSNN(
            in_channels=config.get('in_channels', 2),
            num_classes=config.get('num_classes', 10),
            img_size=config.get('img_size', 34),
            tau=config.get('tau', 2.0),
            T=config.get('T', 20),
        )