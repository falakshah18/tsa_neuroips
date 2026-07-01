# baselines/eprop/eprop_snn.py
"""
Baseline 4: E-prop (Eligibility Propagation)
Online learning rule for SNNs that avoids
the weight transport problem of BPTT.

Core idea:
- Each synapse maintains an eligibility trace
- Learning signal modulates the trace
- No need to store full spike history (online)

Reference:
    Bellec et al. (2020) "A solution to the learning dilemma
    for recurrent networks of spiking neurons"
    Nature Communications.

Biological plausibility score: 4/5
    ✅ Local learning rule
    ✅ Spike-based communication
    ✅ Temporal dynamics
    ✅ No weight transport problem
    ❌ Online learning (partial — uses batches here)
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple, Dict, Optional
import numpy as np


class EligibilityTrace(nn.Module):
    """
    Computes eligibility traces for e-prop.

    Eligibility trace e_ij(t):
        e_ij(t) = h_j(t) * x_i(t-1)

    where:
        h_j(t) = surrogate derivative at neuron j
        x_i(t) = pre-synaptic spike train

    The trace captures the correlation between
    pre-synaptic activity and post-synaptic
    sensitivity — the key e-prop quantity.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau_e: float = 20.0,   # eligibility trace decay
        tau_m: float = 10.0,   # membrane time constant
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.tau_e = tau_e
        self.tau_m = tau_m

        # Decay factors
        self.kappa = np.exp(-1.0 / tau_e)
        self.alpha = np.exp(-1.0 / tau_m)

        # Running eligibility trace
        self.register_buffer(
            'trace',
            torch.zeros(out_features, in_features)
        )

    def reset_trace(self):
        self.trace.zero_()

    def update(
        self,
        pre_spike: torch.Tensor,    # [B, in_features]
        surrogate_grad: torch.Tensor,  # [B, out_features]
    ) -> torch.Tensor:
        """
        Update eligibility trace.

        e(t) = kappa * e(t-1) + h(t) * x(t)

        Args:
            pre_spike: pre-synaptic spikes at t
            surrogate_grad: surrogate gradient at post neuron

        Returns:
            trace: updated eligibility trace [out, in]
        """
        # Outer product: [B, out, in]
        correlation = torch.einsum(
            'bo,bi->boi',
            surrogate_grad,
            pre_spike
        )

        # Average over batch, update trace
        self.trace = (
            self.kappa * self.trace +
            correlation.mean(0)
        )

        return self.trace


class EpropLinear(nn.Module):
    """
    Linear layer with e-prop learning.

    During forward:
        - Computes output spikes
        - Maintains eligibility traces
        - Stores surrogate gradients

    During e-prop update:
        - Uses learning signal L(t)
        - ΔW = η * Σ_t L(t) * e(t)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 10.0,
        tau_e: float = 20.0,
        bias: bool = True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Learnable weights
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        ) if bias else None

        # LIF neuron (membrane potential tracked manually
        # so we can access surrogate gradient)
        self.tau = tau
        self.alpha = np.exp(-1.0 / tau)
        self.threshold = 1.0

        # Surrogate function for gradient
        self.surrogate_fn = surrogate.ATan()

        # Eligibility trace
        self.elig_trace = EligibilityTrace(
            in_features, out_features, tau_e=tau_e
        )

        # Storage for e-prop update
        self.accumulated_dW = None
        self.membrane = None

    def reset(self):
        """Reset membrane potential and eligibility trace."""
        self.membrane = None
        self.elig_trace.reset_trace()
        self.accumulated_dW = torch.zeros_like(self.weight)

    def forward(
        self,
        x: torch.Tensor,  # [B, in_features] at single timestep
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single timestep forward.

        Returns:
            spike: [B, out_features]
            surrogate_grad: [B, out_features]
        """
        B = x.shape[0]
        device = x.device

        # Initialize membrane if needed
        if self.membrane is None:
            self.membrane = torch.zeros(
                B, self.out_features, device=device
            )

        # Linear transform
        input_current = x @ self.weight.t()
        if self.bias is not None:
            input_current += self.bias

        # Membrane update (LIF dynamics)
        self.membrane = self.alpha * self.membrane + input_current

        # Spike generation
        spike = (self.membrane >= self.threshold).float()

        # Surrogate gradient (ATan approximation)
        # d(spike)/d(membrane) ≈ 1/(π(1 + (u-θ)²))
        surrogate_grad = 1.0 / (
            torch.pi * (1.0 + (self.membrane - self.threshold) ** 2)
        )

        # Hard reset
        self.membrane = self.membrane * (1.0 - spike)

        # Update eligibility trace
        self.elig_trace.update(x.detach(), surrogate_grad.detach())

        return spike, surrogate_grad

    def eprop_update(
        self,
        learning_signal: torch.Tensor,  # [B, out_features]
        lr: float = 1e-3,
    ):
        """
        Apply e-prop weight update.

        ΔW_ij = η * L_j * e_ij

        Args:
            learning_signal: error signal per neuron [B, out]
            lr: learning rate
        """
        # Average learning signal over batch
        L = learning_signal.mean(0)  # [out_features]

        # Weight update: L * eligibility trace
        dW = L.unsqueeze(1) * self.elig_trace.trace  # [out, in]

        with torch.no_grad():
            self.weight.data += lr * dW

        # Accumulate for logging
        if self.accumulated_dW is None:
            self.accumulated_dW = dW.detach()
        else:
            self.accumulated_dW += dW.detach()


class EpropSNN(nn.Module):
    """
    Full SNN trained with E-prop.

    Architecture:
        Input → EpropLinear → EpropLinear → EpropLinear → head

    Training loop per batch:
        1. For each timestep t:
            a. Forward through all layers
            b. Compute learning signal from output error
            c. Apply e-prop update to each layer
        2. Train head with standard cross-entropy

    This implements symmetric e-prop (simplified version)
    where learning signal is broadcast to all layers.
    True e-prop uses layer-specific signals but requires
    feedback weights — not implemented here for clarity.
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_size: int = 256,
        num_classes: int = 20,
        tau: float = 10.0,
        tau_e: float = 20.0,
        T: int = 100,
        eprop_lr: float = 1e-3,
    ):
        super().__init__()

        self.T = T
        self.eprop_lr = eprop_lr
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        # E-prop layers
        self.layer1 = EpropLinear(
            input_size, hidden_size,
            tau=tau, tau_e=tau_e,
        )
        self.layer2 = EpropLinear(
            hidden_size, hidden_size,
            tau=tau, tau_e=tau_e,
        )
        self.layer3 = EpropLinear(
            hidden_size, hidden_size // 2,
            tau=tau, tau_e=tau_e,
        )

        # Output head (trained with backprop)
        self.head = nn.Linear(hidden_size // 2, num_classes)

        # Feedback weights for learning signal
        # (simplified: use output weight transpose)
        self.register_buffer(
            'B1',
            torch.randn(hidden_size, num_classes) * 0.01
        )
        self.register_buffer(
            'B2',
            torch.randn(hidden_size, num_classes) * 0.01
        )
        self.register_buffer(
            'B3',
            torch.randn(hidden_size // 2, num_classes) * 0.01
        )

    def reset(self):
        """Reset all layers."""
        self.layer1.reset()
        self.layer2.reset()
        self.layer3.reset()

    def compute_learning_signal(
        self,
        output: torch.Tensor,     # [B, num_classes]
        target: torch.Tensor,     # [B]
        feedback_weights: torch.Tensor,  # [hidden, num_classes]
    ) -> torch.Tensor:
        """
        Compute layer-specific learning signal.

        L = B * (target - output)

        where B are fixed random feedback weights.
        This is the random feedback alignment version
        of e-prop — avoids weight transport problem.

        Returns:
            learning_signal: [B, hidden_size]
        """
        # Error at output
        probs = torch.softmax(output, dim=1)    # [B, C]
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        error = one_hot - probs                  # [B, C]

        # Broadcast to hidden layer via feedback weights
        L = error @ feedback_weights.t()         # [B, hidden]

        return L

    def forward(
        self,
        x: torch.Tensor,             # [T, B, input_size]
        targets: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Full forward pass with optional e-prop updates.

        Args:
            x: [T, B, input_size]
            targets: [B] for e-prop update
            training: apply e-prop if True

        Returns:
            logits: [B, num_classes]
            metrics: dict
        """
        self.reset()

        T, B, _ = x.shape
        device = x.device

        # Accumulate output spikes
        output_spikes = torch.zeros(B, self.num_classes, device=device)
        h3_accum = torch.zeros(B, self.hidden_size // 2, device=device)

        total_spikes = 0
        spike_rates = {'layer1': [], 'layer2': [], 'layer3': []}

        for t in range(T):
            x_t = x[t]  # [B, input_size]

            # Forward through e-prop layers
            s1, sg1 = self.layer1(x_t)    # [B, hidden]
            s2, sg2 = self.layer2(s1)     # [B, hidden]
            s3, sg3 = self.layer3(s2)     # [B, hidden//2]

            # Accumulate
            h3_accum += s3
            total_spikes += s1.sum().item() + s2.sum().item() + s3.sum().item()

            spike_rates['layer1'].append(s1.mean().item())
            spike_rates['layer2'].append(s2.mean().item())
            spike_rates['layer3'].append(s3.mean().item())

            # E-prop updates at each timestep
            if training and targets is not None:
                # Current output estimate
                current_out = self.head(h3_accum / (t + 1))

                # Learning signals for each layer
                L3 = self.compute_learning_signal(
                    current_out, targets, self.B3
                )
                L2 = self.compute_learning_signal(
                    current_out, targets, self.B2
                )
                L1 = self.compute_learning_signal(
                    current_out, targets, self.B1
                )

                # Apply e-prop updates
                self.layer3.eprop_update(L3, lr=self.eprop_lr)
                self.layer2.eprop_update(L2, lr=self.eprop_lr)
                self.layer1.eprop_update(L1, lr=self.eprop_lr)

        # Final classification from averaged hidden state
        h3_mean = h3_accum / T               # [B, hidden//2]
        logits = self.head(h3_mean)          # [B, num_classes]

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (T * B),
            'avg_energy_uJ': total_spikes * 0.1e-12,
            'layer_spike_rates': {
                k: float(np.mean(v))
                for k, v in spike_rates.items()
            },
            'weight_updates': {
                'layer1': self.layer1.accumulated_dW.abs().mean().item()
                if self.layer1.accumulated_dW is not None else 0.0,
                'layer2': self.layer2.accumulated_dW.abs().mean().item()
                if self.layer2.accumulated_dW is not None else 0.0,
                'layer3': self.layer3.accumulated_dW.abs().mean().item()
                if self.layer3.accumulated_dW is not None else 0.0,
            }
        }

        return logits, metrics


class EpropSNN_Vision(nn.Module):
    """
    E-prop SNN for vision datasets.
    Conv features (surrogate grad) + E-prop FC layers.
    Same hybrid approach as SupervisedSTDP_Vision.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        img_size: int = 34,
        tau: float = 10.0,
        tau_e: float = 20.0,
        T: int = 20,
        eprop_lr: float = 1e-3,
    ):
        super().__init__()

        self.T = T
        self.eprop_lr = eprop_lr

        surrogate_fn = surrogate.ATan()

        # Conv feature extractor
        self.features = nn.Sequential(
            layer.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn,
                          detach_reset=True),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(32, 64, 3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(tau=tau, surrogate_function=surrogate_fn,
                          detach_reset=True),
            layer.MaxPool2d(2, 2),
        )

        feature_size = (img_size // 4) ** 2 * 64

        self.flatten = layer.Flatten()

        # E-prop FC layers
        self.eprop1 = EpropLinear(feature_size, 256, tau=tau, tau_e=tau_e)
        self.eprop2 = EpropLinear(256, 128, tau=tau, tau_e=tau_e)

        self.head = nn.Linear(128, num_classes)

        # Feedback weights
        self.register_buffer('B1', torch.randn(256, num_classes) * 0.01)
        self.register_buffer('B2', torch.randn(128, num_classes) * 0.01)

        functional.set_step_mode(self, 'm')

    def reset(self):
        self.eprop1.reset()
        self.eprop2.reset()
        functional.reset_net(self.features)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [T, B, C, H, W]
        """
        self.reset()

        T, B = x.shape[0], x.shape[1]

        # Conv features for all timesteps at once
        feat = self.features(x)         # [T, B, 64, H', W']
        feat = self.flatten(feat)       # [T, B, feature_size]

        h2_accum = torch.zeros(B, 128, device=x.device)
        total_spikes = 0

        for t in range(T):
            feat_t = feat[t]            # [B, feature_size]

            s1, _ = self.eprop1(feat_t) # [B, 256]
            s2, _ = self.eprop2(s1)     # [B, 128]

            h2_accum += s2
            total_spikes += s1.sum().item() + s2.sum().item()

            if training and targets is not None:
                current_out = self.head(h2_accum / (t + 1))

                L2 = self._learning_signal(current_out, targets, self.B2)
                L1 = self._learning_signal(current_out, targets, self.B1)

                self.eprop2.eprop_update(L2, lr=self.eprop_lr)
                self.eprop1.eprop_update(L1, lr=self.eprop_lr)

        h2_mean = h2_accum / T
        logits = self.head(h2_mean)

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (T * B),
            'avg_energy_uJ': total_spikes * 0.1e-12,
        }

        return logits, metrics

    def _learning_signal(self, output, target, feedback):
        probs = torch.softmax(output, dim=1)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, target.unsqueeze(1), 1.0)
        error = one_hot - probs
        return error @ feedback.t()


def get_eprop_model(dataset: str, config: dict) -> nn.Module:
    """Factory function."""
    if dataset == 'shd':
        return EpropSNN(
            input_size=config.get('input_size', 700),
            hidden_size=config.get('hidden_size', 256),
            num_classes=config.get('num_classes', 20),
            tau=config.get('tau', 10.0),
            tau_e=config.get('tau_e', 20.0),
            T=config.get('T', 100),
            eprop_lr=config.get('eprop_lr', 1e-3),
        )
    else:
        return EpropSNN_Vision(
            in_channels=config.get('in_channels', 2),
            num_classes=config.get('num_classes', 10),
            img_size=config.get('img_size', 34),
            tau=config.get('tau', 10.0),
            tau_e=config.get('tau_e', 20.0),
            T=config.get('T', 20),
            eprop_lr=config.get('eprop_lr', 1e-3),
        )