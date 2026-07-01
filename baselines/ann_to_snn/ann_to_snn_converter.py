# baselines/ann_to_snn/ann_to_snn_converter.py
"""
Baseline 2: ANN-to-SNN Conversion
Train a standard ANN, then convert to SNN by:
1. Replacing ReLU with LIF neurons
2. Weight normalization (threshold balancing)
3. No retraining needed after conversion

Reference:
    Diehl et al. (2015) "Fast-Classifying, High-Accuracy SNNs"
    Rueckauer et al. (2017) "Conversion of Continuous-Valued Deep Networks"
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple, Dict
import numpy as np


# ─────────────────────────────────────────────
# STEP 1: Define the ANN to be converted
# ─────────────────────────────────────────────

class SourceANN(nn.Module):
    """
    Standard ANN with ReLU activations.
    This gets trained first, then converted to SNN.
    Uses BatchNorm + ReLU (compatible with conversion).
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        img_size: int = 34,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        feature_size = (img_size // 4) ** 2 * 128

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] standard image input
               Note: for neuromorphic data, average over T first
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class SourceANN_SHD(nn.Module):
    """
    ANN version for SHD dataset.
    """

    def __init__(
        self,
        input_size: int = 700,
        hidden_size: int = 256,
        num_classes: int = 20,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_size] averaged temporal input
        """
        x = self.network(x)
        x = self.head(x)
        return x


# ─────────────────────────────────────────────
# STEP 2: Converted SNN
# ─────────────────────────────────────────────

class ConvertedSNN(nn.Module):
    """
    SNN obtained by converting trained ANN.
    ReLU → LIF neurons with threshold balancing.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 10,
        img_size: int = 34,
        T: int = 100,
        thresholds: Dict = None,
    ):
        super().__init__()

        self.T = T
        self.thresholds = thresholds or {}

        # Same architecture but with LIF instead of ReLU
        self.features = nn.Sequential(
            layer.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(32),
            neuron.LIFNode(
                tau=float('inf'),           # IF neuron (no leak)
                v_threshold=self.thresholds.get('features.2', 1.0),
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            ),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.LIFNode(
                tau=float('inf'),
                v_threshold=self.thresholds.get('features.6', 1.0),
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            ),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.LIFNode(
                tau=float('inf'),
                v_threshold=self.thresholds.get('features.10', 1.0),
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            ),
        )

        feature_size = (img_size // 4) ** 2 * 128

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(feature_size, 512),
            neuron.LIFNode(
                tau=float('inf'),
                v_threshold=self.thresholds.get('classifier.1', 1.0),
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            ),
            layer.Dropout(0.5),
            layer.Linear(512, num_classes),
        )

        functional.set_step_mode(self, 'm')

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [T, B, C, H, W]
        Returns:
            logits: [B, num_classes]
            metrics: dict
        """
        x = self.features(x)
        x = self.classifier(x)

        # In SpikingJelly activation_based API (multi-step mode), the LIF node
        # returns spike tensors through forward() — there is no .spike attribute.
        # We track spikes directly from the output tensor x before mean-pooling.
        # x at this point is [T, B, num_classes] — count non-zero entries.
        total_spikes = (x > 0).float().sum().item()

        out = x.mean(0)

        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (self.T * out.shape[0]),
            'avg_energy_uJ': total_spikes * 0.1e-12,
        }

        return out, metrics

    def reset(self):
        functional.reset_net(self)


# ─────────────────────────────────────────────
# STEP 3: Converter
# ─────────────────────────────────────────────

class ANNtoSNNConverter:
    """
    Handles the full ANN → SNN conversion pipeline:
    1. Train ANN on averaged neuromorphic data
    2. Compute activation percentiles per layer
    3. Set thresholds via threshold balancing
    4. Transfer weights to ConvertedSNN
    5. Evaluate with T timesteps
    """

    def __init__(self, percentile: float = 99.9):
        self.percentile = percentile
        self.activation_stats = {}
        self.hooks = []

    def _register_hooks(self, ann: nn.Module):
        """
        Register forward hooks to collect
        activation statistics per layer.
        """
        self.activation_stats = {}

        def make_hook(name):
            def hook(module, input, output):
                if name not in self.activation_stats:
                    self.activation_stats[name] = []
                self.activation_stats[name].append(
                    output.detach().cpu().flatten()
                )
            return hook

        for name, module in ann.named_modules():
            if isinstance(module, nn.ReLU):
                h = module.register_forward_hook(make_hook(name))
                self.hooks.append(h)

    def _remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def compute_thresholds(
        self,
        ann: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
        n_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Threshold balancing:
        Set each layer's threshold to the Nth percentile
        of its activations on calibration data.
        """
        ann.eval()
        self._register_hooks(ann)

        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= n_batches:
                    break

                # For neuromorphic data average over T
                if data.dim() == 5:
                    data = data.mean(1)  # [B, C, H, W] — average over T dim

                data = data.to(device)
                ann(data)

        self._remove_hooks()

        # Compute percentile thresholds
        thresholds = {}
        for name, activations in self.activation_stats.items():
            all_acts = torch.cat(activations)
            threshold = float(
                torch.quantile(all_acts, self.percentile / 100.0)
            )
            thresholds[name] = max(threshold, 1e-6)  # avoid zero

        print(f"Computed thresholds for {len(thresholds)} layers:")
        for name, thresh in thresholds.items():
            print(f"  {name}: {thresh:.4f}")

        return thresholds

    def transfer_weights(
        self,
        ann: nn.Module,
        snn: nn.Module,
    ):
        """
        Copy weights from ANN to SNN.
        Only copies Conv2d and Linear weights
        (neuron layers have no shared weights).
        """
        ann_state = ann.state_dict()
        snn_state = snn.state_dict()

        transferred = 0
        for key in snn_state:
            if key in ann_state:
                if ann_state[key].shape == snn_state[key].shape:
                    snn_state[key] = ann_state[key]
                    transferred += 1

        snn.load_state_dict(snn_state)
        print(f"Transferred {transferred} weight tensors from ANN to SNN")

    def convert(
        self,
        ann: nn.Module,
        snn_class,
        snn_kwargs: dict,
        dataloader: torch.utils.data.DataLoader,
        device: str = 'cuda',
    ) -> nn.Module:
        """
        Full conversion pipeline.

        Args:
            ann: trained ANN
            snn_class: ConvertedSNN class
            snn_kwargs: constructor args for SNN
            dataloader: calibration data for threshold computation
            device: cuda or cpu

        Returns:
            converted SNN ready for evaluation
        """
        print("\nStarting ANN → SNN conversion...")

        # Step 1: Compute thresholds
        print("\nStep 1: Computing activation thresholds...")
        thresholds = self.compute_thresholds(ann, dataloader, device)

        # Step 2: Build SNN with computed thresholds
        print("\nStep 2: Building SNN with balanced thresholds...")
        snn = snn_class(**snn_kwargs, thresholds=thresholds)
        snn = snn.to(device)

        # Step 3: Transfer weights
        print("\nStep 3: Transferring weights...")
        self.transfer_weights(ann, snn)

        print("\nConversion complete.")
        return snn


def get_ann_to_snn_model(dataset: str, config: dict, device: str = 'cuda'):
    """
    Factory: returns (ann, snn_class, snn_kwargs) tuple for both datasets.
    Train ANN first, then call converter.convert().
    """
    if dataset == 'shd':
        ann = SourceANN_SHD(
            input_size=config.get('input_size', 700),
            hidden_size=config.get('hidden_size', 256),
            num_classes=config.get('num_classes', 20),
        )
        # SHD uses FC-only ANN; no ConvertedSNN vision class needed
        # Return None for snn_class — caller uses ann directly after training
        return ann, None, {}
    else:
        ann = SourceANN(
            in_channels=config.get('in_channels', 2),
            num_classes=config.get('num_classes', 10),
            img_size=config.get('img_size', 34),
        )
        snn_kwargs = {
            'in_channels': config.get('in_channels', 2),
            'num_classes': config.get('num_classes', 10),
            'img_size': config.get('img_size', 34),
            'T': config.get('T', 100),
        }
        return ann, ConvertedSNN, snn_kwargs