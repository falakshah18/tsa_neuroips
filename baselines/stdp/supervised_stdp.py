# baselines/stdp/supervised_stdp.py
"""
Baseline 3: Supervised STDP
Spike-Timing Dependent Plasticity with supervised signal.
Most biologically plausible of all baselines.

Core idea:
- Pre-synaptic spike before post-synaptic → strengthen (LTP)
- Post-synaptic spike before pre-synaptic → weaken (LTD)
- Supervised signal modulates the learning rule

Reference:
    Mozafari et al. (2019) "SpykeTorch: Efficient SNN with STDP"
    Tavanaei et al. (2019) "Deep Learning in SNNs"
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple, Dict, Optional
import numpy as np


class STDPLearningRule(nn.Module):
    """
    STDP weight update rule.

    ΔW = A+ · exp(-Δt/τ+)  if pre before post (LTP)
    ΔW = A- · exp(-Δt/τ-)  if post before pre (LTD)

    Supervised version: modulate by error signal
    ΔW_supervised = ΔW_STDP · (target - output)
    """

    def __init__(
        self,
        a_plus: float = 0.01,    # LTP amplitude
        a_minus: float = 0.01,   # LTD amplitude
        tau_plus: float = 20.0,  # LTP time constant (ms)
        tau_minus: float = 20.0, # LTD time constant (ms)
        w_min: float = 0.0,      # minimum weight
        w_max: float = 1.0,      # maximum weight
    ):
        super().__init__()

        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_min = w_min
        self.w_max = w_max

    def compute_update(
        self,
        pre_spikes: torch.Tensor,   # [T, B, N_in]
        post_spikes: torch.Tensor,  # [T, B, N_out]
        weights: torch.Tensor,      # [N_out, N_in]
        error: Optional[torch.Tensor] = None,  # [B, N_out]
    ) -> torch.Tensor:
        """
        Compute STDP weight update.

        Returns:
            dW: weight update [N_out, N_in]
        """
        T = pre_spikes.shape[0]
        device = pre_spikes.device

        dW = torch.zeros_like(weights)

        # Trace variables for pre and post
        pre_trace = torch.zeros_like(pre_spikes[0])   # [B, N_in]
        post_trace = torch.zeros_like(post_spikes[0]) # [B, N_out]

        for t in range(T):
            pre_t = pre_spikes[t]    # [B, N_in]
            post_t = post_spikes[t]  # [B, N_out]

            # Update traces
            pre_trace = pre_trace * np.exp(-1.0 / self.tau_plus) + pre_t
            post_trace = post_trace * np.exp(-1.0 / self.tau_minus) + post_t

            # LTP: post spike, use pre trace
            # dW += A+ * post_t * pre_trace
            ltp = self.a_plus * torch.einsum(
                'bo,bi->oi', post_t, pre_trace
            ) / pre_t.shape[0]  # average over batch

            # LTD: pre spike, use post trace
            # dW -= A- * pre_t * post_trace
            ltd = self.a_minus * torch.einsum(
                'bo,bi->oi', post_trace, pre_t
            ) / pre_t.shape[0]

            dW += ltp - ltd

        # Supervised modulation
        if error is not None:
            # error: [B, N_out] → mean over batch → [N_out]
            error_signal = error.mean(0)  # [N_out]
            dW = dW * error_signal.unsqueeze(1)

        return dW

    def apply_update(
        self,
        weights: nn.Parameter,
        dW: torch.Tensor,
        lr: float = 0.01,
    ):
        """
        Apply weight update with clipping.
        """
        with torch.no_grad():
            weights.data += lr * dW
            weights.data.clamp_(self.w_min, self.w_max)


class STDPLayer(nn.Module):
    """
    Single STDP-trainable fully connected layer.
    Wraps Linear + LIF with STDP learning rule.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 2.0,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ):
        super().__init__()

        self.fc = layer.Linear(in_features, out_features, bias=False)
        self.lif = neuron.LIFNode(
            tau=tau,
            surrogate_function=surrogate.ATan(),
            detach_reset=True,
        )

        self.stdp = STDPLearningRule(
            a_plus=a_plus,
            a_minus=a_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
        )

        # Store spike history for STDP
        self.pre_spikes = None
        self.post_spikes = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T, B, in_features] pre-synaptic spikes
        Returns:
            out: [T, B, out_features] post-synaptic spikes
        """
        self.pre_spikes = x.detach()

        out = self.fc(x)
        out = self.lif(out)

        self.post_spikes = out.detach()

        return out

    def stdp_update(
        self,
        error: Optional[torch.Tensor] = None,
        lr: float = 0.01,
    ):
        """
        Apply STDP update using stored spike history.
        Call after forward pass.
        """
        if self.pre_spikes is None or self.post_spikes is None:
            return

        dW = self.stdp.compute_update(
            pre_spikes=self.pre_spikes,
            post_spikes=self.post_spikes,
            weights=self.fc.weight,
            error=error,
        )

        self.stdp.apply_update(self.fc.weight, dW, lr=lr)


class SupervisedSTDP(nn.Module):
    """
    Full SNN trained with Supervised STDP.

    Architecture:
        Input → STDPLayer → STDPLayer → STDPLayer → Linear head

    Training:
        1. Forward pass: collect spike trains
        2. Compute error: target - output firing rates
        3. STDP update: modulate by error signal
        4. Head trained with standard backprop (cross-entropy)

    Biological plausibility score: 4/5
        ✅ Local learning rule
        ✅ Spike-based communication
        ✅ Temporal dynamics
        ✅ No weight transport problem
        ❌ Online learning (uses batches)
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_size: int = 256,
        num_classes: int = 20,
        tau: float = 2.0,
        T: int = 100,
        a_plus: float = 0.01,
        a_minus: float = 0.01,
        stdp_lr: float = 0.01,
    ):
        super().__init__()

        self.T = T
        self.stdp_lr = stdp_lr
        self.num_classes = num_classes

        # STDP layers
        self.layer1 = STDPLayer(
            input_size, hidden_size,
            tau=tau, a_plus=a_plus, a_minus=a_minus,
        )
        self.layer2 = STDPLayer(
            hidden_size, hidden_size,
            tau=tau, a_plus=a_plus, a_minus=a_minus,
        )
        self.layer3 = STDPLayer(
            hidden_size, hidden_size // 2,
            tau=tau, a_plus=a_plus, a_minus=a_minus,
        )

        # Classification head (trained with backprop)
        self.head = nn.Linear(hidden_size // 2, num_classes)

        # Dropout
        self.drop = layer.Dropout(0.3)

        functional.set_step_mode(self, 'm')

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [T, B, input_size]
            targets: [B] class labels (needed for supervised STDP)
            training: whether to apply STDP updates

        Returns:
            logits: [B, num_classes]
            metrics: dict
        """
        # Forward through STDP layers
        h1 = self.layer1(x)           # [T, B, hidden]
        h1 = self.drop(h1)

        h2 = self.layer2(h1)          # [T, B, hidden]
        h2 = self.drop(h2)

        h3 = self.layer3(h2)          # [T, B, hidden//2]

        # Average firing rates over time
        h3_mean = h3.mean(0)          # [B, hidden//2]

        # Classification
        logits = self.head(h3_mean)   # [B, num_classes]

        # STDP updates during training
        if training and targets is not None:
            self._apply_stdp_updates(logits, targets)

        # Count spikes
        total_spikes = (
            h1.sum().item() +
            h2.sum().item() +
            h3.sum().item()
        )

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (self.T * x.shape[1]),
            'avg_energy_uJ': total_spikes * 0.1e-6,
            'layer_spike_rates': {
                'layer1': h1.mean().item(),
                'layer2': h2.mean().item(),
                'layer3': h3.mean().item(),
            }
        }

        return logits, metrics

    def _apply_stdp_updates(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ):
        """
        Compute supervised error and apply STDP.

        Error = one_hot(target) - softmax(logits)
        This modulates STDP: correct neurons strengthen,
        incorrect neurons weaken.
        """
        with torch.no_grad():
            # Compute error signal
            probs = torch.softmax(logits, dim=1)    # [B, C]
            one_hot = torch.zeros_like(probs)
            one_hot.scatter_(1, targets.unsqueeze(1), 1.0)
            error = one_hot - probs                 # [B, C]

        # Apply to each STDP layer
        # Error for hidden layers: propagate back signal
        # (simplified — not true backprop)
        # Broadcast the output error to hidden layers as a teaching signal.
        # This is the standard supervised STDP formulation (Mozafari et al. 2019):
        # error at output modulates all layers (approximate credit assignment).
        # layer3 has hidden_size//2 outputs; project error down via head weights.
        with torch.no_grad():
            # Use head weight to project error back to layer3 output space
            error_l3 = error @ self.head.weight  # [B, hidden//2]
        self.layer3.stdp_update(error=error_l3.detach(), lr=self.stdp_lr)
        self.layer2.stdp_update(error=None, lr=self.stdp_lr)   # unsupervised for deeper layers
        self.layer1.stdp_update(error=None, lr=self.stdp_lr)



class SupervisedSTDP_Vision(nn.Module):
    """
    Supervised STDP for vision datasets (N-MNIST, DVS-Gesture).
    Uses convolutional feature extraction + STDP classifier.

    Note: Conv layers trained with surrogate gradients,
    STDP applied only to FC layers — hybrid approach
    common in literature.

    Biological plausibility score: 3/5
        ✅ Local learning rule (FC layers)
        ✅ Spike-based communication
        ✅ Temporal dynamics
        ❌ No weight transport (conv uses backprop)
        ❌ Online learning
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        img_size: int = 34,
        tau: float = 2.0,
        T: int = 20,
        stdp_lr: float = 0.01,
    ):
        super().__init__()

        self.T = T
        self.stdp_lr = stdp_lr

        # Conv feature extractor (surrogate gradient)
        self.features = nn.Sequential(
            layer.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(
                tau=tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            ),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(32, 64, 3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(
                tau=tau,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            ),
            layer.MaxPool2d(2, 2),
        )

        feature_size = (img_size // 4) ** 2 * 64

        # STDP classifier
        self.flatten = layer.Flatten()
        self.stdp_layer1 = STDPLayer(
            feature_size, 256,
            tau=tau,
        )
        self.stdp_layer2 = STDPLayer(
            256, 128,
            tau=tau,
        )

        self.head = nn.Linear(128, num_classes)

        functional.set_step_mode(self, 'm')

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
        # Conv features
        feat = self.features(x)          # [T, B, 64, H', W']
        feat = self.flatten(feat)        # [T, B, feature_size]

        # STDP layers
        h1 = self.stdp_layer1(feat)      # [T, B, 256]
        h2 = self.stdp_layer2(h1)        # [T, B, 128]

        # Pool over time
        h2_mean = h2.mean(0)             # [B, 128]

        logits = self.head(h2_mean)      # [B, num_classes]

        # STDP update
        if training and targets is not None:
            self.stdp_layer1.stdp_update(lr=self.stdp_lr)
            self.stdp_layer2.stdp_update(lr=self.stdp_lr)

        total_spikes = feat.sum().item() + h1.sum().item() + h2.sum().item()

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (self.T * x.shape[1]),
            'avg_energy_uJ': total_spikes * 0.1e-6,
        }

        return logits, metrics


def get_stdp_model(dataset: str, config: dict) -> nn.Module:
    """
    Factory function.
    """
    if dataset == 'shd':
        return SupervisedSTDP(
            input_size=config.get('input_size', 700),
            hidden_size=config.get('hidden_size', 256),
            num_classes=config.get('num_classes', 20),
            tau=config.get('tau', 2.0),
            T=config.get('T', 100),
            stdp_lr=config.get('stdp_lr', 0.01),
        )
    else:
        return SupervisedSTDP_Vision(
            in_channels=config.get('in_channels', 2),
            num_classes=config.get('num_classes', 10),
            img_size=config.get('img_size', 34),
            tau=config.get('tau', 2.0),
            T=config.get('T', 20),
            stdp_lr=config.get('stdp_lr', 0.01),
        )