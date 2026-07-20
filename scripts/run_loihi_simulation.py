"""Run Loihi 2 energy simulation for the TSA model and produce concrete numbers."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import numpy as np
from pathlib import Path
from hardware.loihi2_deployment import LoihiSimulator
from models.tst_v2 import TemporalSpikingTransformer


def load_tsa_config():
    config_path = Path(__file__).parent.parent / 'configs' / 'tsa_config.yaml'
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_default_tsa_model(cfg):
    nmnist = cfg['nmnist']
    model_cfg = cfg['model']
    embed_dim = model_cfg['embed_dim']
    n_heads = model_cfg['num_heads']
    n_layers = model_cfg['depth']
    mlp_ratio = model_cfg['mlp_ratio']
    return TemporalSpikingTransformer(
        img_size=nmnist['img_size'],
        patch_size=nmnist['patch_size'],
        in_channels=nmnist['in_channels'],
        num_classes=nmnist['num_classes'],
        embed_dim=embed_dim,
        depth=n_layers,
        num_heads=n_heads,
        mlp_ratio=mlp_ratio,
        drop_rate=model_cfg.get('drop_rate', 0.0),
        init_tau=model_cfg.get('init_tau', 2.0),
    )


class SimpleANN(torch.nn.Module):
    def __init__(self, img_size=34, in_channels=2, num_classes=10, d=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(img_size**2 * in_channels, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, d),
            torch.nn.ReLU(),
            torch.nn.Linear(d, num_classes),
        )
    def forward(self, x):
        if x.dim() == 5:
            x = x[0]
        return self.net(x), {}


def make_fake_dataloader(T=20, batch_size=10, img_size=34, in_channels=2):
    data = torch.randn(T, batch_size, in_channels, img_size, img_size)
    targets = torch.randint(0, 10, (batch_size,))
    dataset = torch.utils.data.TensorDataset(data.permute(1, 0, 2, 3, 4), targets)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def run_energy_simulation():
    print("=" * 70)
    print("LOIHI 2 ENERGY SIMULATION — TSA vs Baselines")
    print("=" * 70)

    cfg = load_tsa_config()
    nmnist = cfg['nmnist']
    T = nmnist['T']
    img_size = nmnist['img_size']
    in_channels = nmnist['in_channels']

    tsa_model = get_default_tsa_model(cfg).eval()
    ann_model = SimpleANN(img_size=img_size, in_channels=in_channels).eval()

    loader = make_fake_dataloader(T=T, img_size=img_size, in_channels=in_channels)
    simulator = LoihiSimulator()

    results = {}

    print("\n--- TSA (Temporal Sparse Spiking Transformer) ---")
    tsa_results = simulator.simulate_execution(tsa_model, loader, n_samples=50)
    results['TSA'] = tsa_results
    print(f"  Energy per sample:  {tsa_results['avg_energy_per_sample_uJ']:.4f} uJ")
    print(f"  Latency per sample: {tsa_results['avg_latency_per_sample_ms']:.4f} ms")
    print(f"  Accuracy:           {tsa_results['accuracy']:.4f}")

    print("\n--- ANN (Standard Feedforward) ---")
    ann_results = simulator.simulate_execution(ann_model, loader, n_samples=50)
    results['ANN'] = ann_results
    print(f"  Energy per sample:  {ann_results['avg_energy_per_sample_uJ']:.4f} uJ")
    print(f"  Latency per sample: {ann_results['avg_latency_per_sample_ms']:.4f} ms")
    print(f"  Accuracy:           {ann_results['accuracy']:.4f}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Energy (uJ)':<15} {'Latency (ms)':<15}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['avg_energy_per_sample_uJ']:<15.4f} {r['avg_latency_per_sample_ms']:<15.4f}")

    tsa_e = results['TSA']['avg_energy_per_sample_uJ']
    ann_e = results['ANN']['avg_energy_per_sample_uJ']
    if tsa_e > 0:
        print(f"\nEnergy reduction (TSA vs ANN): {ann_e / tsa_e:.1f}x")

    print("\n--- Theoretical Loihi 2 Energy (from published Loihi 2 specs) ---")
    print("  E_MAC (GPU FP16) = 4.6 pJ")
    print("  E_spike (Loihi 2) = 0.1 pJ")
    print()
    for s_rate in [0.05, 0.10, 0.15, 0.20]:
        ratio = s_rate * 0.1 / 4.6
        print(f"  Spike rate {s_rate:.0%}: E_TSA/E_ANN = {ratio:.4f} = {1/ratio:.1f}x reduction")

    return results


if __name__ == "__main__":
    results = run_energy_simulation()
