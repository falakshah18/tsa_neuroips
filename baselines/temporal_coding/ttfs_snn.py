# baselines/temporal_coding/ttfs_snn.py
"""
Baseline 5: Time-To-First-Spike (TTFS) Temporal Coding
Information encoded in WHEN neurons spike, not HOW MANY times.

Core idea:
    - Earlier spike = stronger activation
    - Each neuron fires AT MOST once per input
    - More energy efficient than rate coding
    - Latency encoding: x_i → t_i = (1 - x_i) * T

Reference:
    Thorpe et al. (2001) "Spike-based strategies for rapid processing"
    Rueckauer & Liu (2018) "Conversion of ANN to TTFS SNN"
    Park et al. (2020) "T2FSNN: Deep TTFS SNNs"

Biological plausibility score: 3/5
    ✅ Spike-based communication
    ✅ Temporal dynamics
    ✅ Energy efficient (one spike per neuron)
    ❌ Local learning rule (uses backprop variant)
    ❌ Not fully online
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple, Dict, Optional
import numpy as np


# ─────────────────────────────────────────────
# TTFS Encoding
# ─────────────────────────────────────────────

class TTFSEncoder(nn.Module):
    """
    Encodes input values as spike times.

    Larger input value → earlier spike time
    Smaller input value → later spike time (or no spike)

    Encoding: t_spike = (1 - x) * T
    where x ∈ [0, 1] is normalized input

    Creates spike train [T, B, N] where each
    neuron fires exactly once at its encoded time.
    """

    def __init__(self, T: int = 20):
        super().__init__()
        self.T = T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N] or [B, C, H, W] normalized input
               Values should be in [0, 1]

        Returns:
            spikes: [T, B, N] TTFS encoded spike train
        """
        # Flatten spatial dims if needed
        if x.dim() > 2:
            B = x.shape[0]
            x = x.flatten(1)  # [B, N]

        B, N = x.shape
        device = x.device

        # Clamp to [0, 1]
        x = x.clamp(0.0, 1.0)

        # Compute spike times: earlier = stronger
        # t_spike ∈ {0, 1, ..., T-1}
        spike_times = ((1.0 - x) * (self.T - 1)).long()  # [B, N]

        # Build spike train
        spikes = torch.zeros(self.T, B, N, device=device)

        for t in range(self.T):
            # Neuron fires if its spike time == t
            spikes[t] = (spike_times == t).float()

        return spikes


class TTFSEncoderNeuromorphic(nn.Module):
    """
    TTFS encoder for neuromorphic data [T, B, C, H, W].
    Converts event frames to TTFS representation.

    For neuromorphic data:
    - Already temporal, so we find first spike time per pixel
    - Re-encode as clean TTFS signal
    """

    def __init__(self, T_out: int = 20):
        super().__init__()
        self.T_out = T_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [T_in, B, C, H, W] neuromorphic frames

        Returns:
            spikes: [T_out, B, C*H*W] TTFS encoded
        """
        T_in, B, C, H, W = x.shape
        N = C * H * W
        device = x.device

        # Find first spike time for each pixel
        # x[t] > 0 means event at time t
        x_flat = x.reshape(T_in, B, N)  # [T_in, B, N]

        # First nonzero time per neuron
        # If never fires, set to T_in (no spike)
        has_spike = (x_flat > 0)  # [T_in, B, N]

        # For each (B, N), find first T where has_spike is True
        # Use argmax on has_spike (returns first True index)
        first_spike = torch.where(
            has_spike.any(0),
            has_spike.float().argmax(0),
            torch.full((B, N), T_in, device=device, dtype=torch.long)
        )  # [B, N]

        # Normalize to [0, T_out]
        first_spike_norm = (first_spike.float() / T_in * self.T_out).long()
        first_spike_norm = first_spike_norm.clamp(0, self.T_out - 1)

        # Build output spike train
        spikes = torch.zeros(self.T_out, B, N, device=device)
        for t in range(self.T_out):
            spikes[t] = (first_spike_norm == t).float()

        return spikes


# ─────────────────────────────────────────────
# TTFS Neuron
# ─────────────────────────────────────────────

class TTFSNeuron(nn.Module):
    """
    LIF neuron that fires at most once (TTFS constraint).

    Once fired, neuron is inhibited for rest of window.
    This enforces the TTFS coding scheme.

    Gradient approximation:
        Since spike time is discrete, use continuous
        relaxation for training (soft TTFS).
    """

    def __init__(
        self,
        tau: float = 2.0,
        threshold: float = 1.0,
        inhibit_after_spike: bool = True,
    ):
        super().__init__()

        self.tau = tau
        self.threshold = threshold
        self.inhibit_after_spike = inhibit_after_spike
        self.surrogate_fn = surrogate.ATan()
        self.decay = np.exp(-1.0 / self.tau)

        # Track if neuron has already fired
        self.has_fired = None
        self.membrane = None

    def reset(self):
        self.has_fired = None
        self.membrane = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single timestep forward.

        Args:
            x: [B, N] input current

        Returns:
            spike: [B, N]
        """
        B, N = x.shape
        device = x.device

        # Initialize
        if self.membrane is None:
            self.membrane = torch.zeros(B, N, device=device)
        if self.has_fired is None:
            self.has_fired = torch.zeros(B, N, device=device)

        # Update membrane (only for neurons that haven't fired)
        if self.inhibit_after_spike:
            active = (1.0 - self.has_fired)
        else:
            active = torch.ones_like(self.has_fired)

        self.membrane = self.decay * self.membrane + x * active

        # Spike generation with surrogate gradient
        spike = self.surrogate_fn(self.membrane - self.threshold)

        # Hard threshold for actual spike
        spike_hard = (self.membrane >= self.threshold).float()

        # Inhibit fired neurons
        if self.inhibit_after_spike:
            self.has_fired = torch.clamp(
                self.has_fired + spike_hard, 0.0, 1.0
            )

        # Use hard spike but with surrogate gradient
        spike = spike_hard + (spike - spike.detach())

        # Hard reset
        self.membrane = self.membrane * (1.0 - spike_hard)

        return spike


# ─────────────────────────────────────────────
# TTFS Network
# ─────────────────────────────────────────────

class TTFSLayer(nn.Module):
    """
    Single TTFS layer: Linear + TTFSNeuron.
    Processes one timestep at a time.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: float = 2.0,
        threshold: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()

        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.neuron = TTFSNeuron(
            tau=tau,
            threshold=threshold,
            inhibit_after_spike=True,
        )

    def reset(self):
        self.neuron.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_features] single timestep
        Returns:
            spike: [B, out_features]
        """
        current = self.fc(x)
        spike = self.neuron(current)
        return spike


class TTFSNetwork(nn.Module):
    """
    Full TTFS SNN for temporal classification.

    Pipeline:
        Input → TTFS Encode → TTFSLayer x3 → Decode → Classify

    Decoding:
        Convert output spike times back to values:
        x_i = 1 - (t_spike_i / T)
        Earlier spike → higher value → stronger evidence

    Training:
        Standard backprop through surrogate gradients.
        The TTFS constraint is enforced via inhibition,
        not through the loss function.
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_size: int = 256,
        num_classes: int = 20,
        tau: float = 2.0,
        T: int = 100,
        T_encode: int = 20,
    ):
        super().__init__()

        self.T = T
        self.T_encode = T_encode
        self.num_classes = num_classes

        # Encoder
        self.encoder = TTFSEncoder(T=T_encode)

        # TTFS layers
        self.layer1 = TTFSLayer(input_size, hidden_size, tau=tau)
        self.layer2 = TTFSLayer(hidden_size, hidden_size, tau=tau)
        self.layer3 = TTFSLayer(hidden_size, hidden_size // 2, tau=tau)

        # Output head
        self.head = nn.Linear(hidden_size // 2, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def decode_spike_times(
        self,
        spike_train: torch.Tensor,  # [T, B, N]
    ) -> torch.Tensor:
        """
        Convert spike train to rate-like values.

        For TTFS: earlier spike = higher value
        t_first = first spike time (or T if no spike)
        value = 1 - (t_first / T)

        Returns:
            values: [B, N]
        """
        T, B, N = spike_train.shape
        device = spike_train.device

        has_spike = spike_train.any(0)  # [B, N]

        # First spike time
        first_spike_time = torch.where(
            has_spike,
            spike_train.float().argmax(0).float(),
            torch.full((B, N), float(T), device=device)
        )

        # Decode to value
        values = 1.0 - (first_spike_time / T)
        values = values * has_spike.float()  # zero if no spike

        return values

    def reset(self):
        self.layer1.reset()
        self.layer2.reset()
        self.layer3.reset()

    def forward(
        self,
        x: torch.Tensor,  # [T_in, B, input_size] or [B, input_size]
        targets: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [T, B, input_size] temporal input

        Returns:
            logits: [B, num_classes]
            metrics: dict
        """
        self.reset()

        # If input is temporal, take mean for encoding
        if x.dim() == 3:
            x_mean = x.mean(0)  # [B, input_size]
        else:
            x_mean = x          # [B, input_size]

        # TTFS encode
        encoded = self.encoder(x_mean)  # [T_encode, B, input_size]

        T_enc = encoded.shape[0]
        B = encoded.shape[1]
        device = x.device

        # Spike trains per layer
        spikes1 = []
        spikes2 = []
        spikes3 = []

        for t in range(T_enc):
            x_t = encoded[t]              # [B, input_size]

            s1 = self.layer1(x_t)         # [B, hidden]
            s1 = self.dropout(s1)

            s2 = self.layer2(s1)          # [B, hidden]
            s2 = self.dropout(s2)

            s3 = self.layer3(s2)          # [B, hidden//2]

            spikes1.append(s1)
            spikes2.append(s2)
            spikes3.append(s3)

        # Stack spike trains
        spikes1 = torch.stack(spikes1, dim=0)  # [T, B, hidden]
        spikes2 = torch.stack(spikes2, dim=0)
        spikes3 = torch.stack(spikes3, dim=0)  # [T, B, hidden//2]

        # Decode spike times to values
        decoded = self.decode_spike_times(spikes3)  # [B, hidden//2]

        # Classify
        logits = self.head(decoded)               # [B, num_classes]

        # Count spikes (TTFS should be very sparse)
        total_spikes = (
            spikes1.sum().item() +
            spikes2.sum().item() +
            spikes3.sum().item()
        )

        # Theoretical max spikes (one per neuron per sample)
        max_possible = B * (
            spikes1.shape[2] +
            spikes2.shape[2] +
            spikes3.shape[2]
        )

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / max(max_possible, 1),
            'avg_energy_uJ': total_spikes * 0.1e-12,
            'sparsity': 1.0 - (total_spikes / max(max_possible, 1)),
            'layer_spike_counts': {
                'layer1': spikes1.sum().item(),
                'layer2': spikes2.sum().item(),
                'layer3': spikes3.sum().item(),
            }
        }

        return logits, metrics


class TTFSNetwork_Vision(nn.Module):
    """
    TTFS SNN for vision datasets (N-MNIST, DVS-Gesture).

    Pipeline:
        [T, B, C, H, W]
        → TTFSEncoderNeuromorphic
        → [T_enc, B, C*H*W]
        → Conv TTFS layers
        → TTFSLayer FC
        → Classify
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        img_size: int = 34,
        tau: float = 2.0,
        T: int = 20,
        T_encode: int = 20,
    ):
        super().__init__()

        self.T = T
        self.T_encode = T_encode
        self.img_size = img_size
        self.in_channels = in_channels

        # Neuromorphic TTFS encoder
        self.encoder = TTFSEncoderNeuromorphic(T_out=T_encode)

        input_size = in_channels * img_size * img_size

        # TTFS FC layers
        # (Conv TTFS is complex; use FC for clarity)
        self.layer1 = TTFSLayer(input_size, 512, tau=tau)
        self.layer2 = TTFSLayer(512, 256, tau=tau)
        self.layer3 = TTFSLayer(256, 128, tau=tau)

        self.head = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.3)

    def decode_spike_times(
        self,
        spike_train: torch.Tensor,
    ) -> torch.Tensor:
        T, B, N = spike_train.shape
        device = spike_train.device

        has_spike = spike_train.any(0)
        first_spike_time = torch.where(
            has_spike,
            spike_train.float().argmax(0).float(),
            torch.full((B, N), float(T), device=device)
        )
        values = 1.0 - (first_spike_time / T)
        values = values * has_spike.float()
        return values

    def reset(self):
        self.layer1.reset()
        self.layer2.reset()
        self.layer3.reset()

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

        # TTFS encode neuromorphic data
        encoded = self.encoder(x)  # [T_enc, B, C*H*W]

        T_enc, B, N = encoded.shape

        spikes1, spikes2, spikes3 = [], [], []

        for t in range(T_enc):
            x_t = encoded[t]              # [B, N]

            s1 = self.layer1(x_t)
            s1 = self.dropout(s1)

            s2 = self.layer2(s1)
            s2 = self.dropout(s2)

            s3 = self.layer3(s2)

            spikes1.append(s1)
            spikes2.append(s2)
            spikes3.append(s3)

        spikes1 = torch.stack(spikes1, dim=0)
        spikes2 = torch.stack(spikes2, dim=0)
        spikes3 = torch.stack(spikes3, dim=0)

        decoded = self.decode_spike_times(spikes3)
        logits = self.head(decoded)

        total_spikes = (
            spikes1.sum().item() +
            spikes2.sum().item() +
            spikes3.sum().item()
        )

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / max(
                B * (spikes1.shape[2] + spikes2.shape[2] + spikes3.shape[2]), 1
            ),
            'avg_energy_uJ': total_spikes * 0.1e-12,
            'sparsity': 1.0 - total_spikes / max(
                B * (spikes1.shape[2] + spikes2.shape[2] + spikes3.shape[2]), 1
            ),
        }

        return logits, metrics


def get_ttfs_model(dataset: str, config: dict) -> nn.Module:
    """Factory function."""
    if dataset == 'shd':
        return TTFSNetwork(
            input_size=config.get('input_size', 700),
            hidden_size=config.get('hidden_size', 256),
            num_classes=config.get('num_classes', 20),
            tau=config.get('tau', 2.0),
            T=config.get('T', 100),
            T_encode=config.get('T_encode', 20),
        )
    else:
        return TTFSNetwork_Vision(
            in_channels=config.get('in_channels', 2),
            num_classes=config.get('num_classes', 10),
            img_size=config.get('img_size', 34),
            tau=config.get('tau', 2.0),
            T=config.get('T', 20),
            T_encode=config.get('T_encode', 20),
        )