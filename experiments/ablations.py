# experiments/ablations.py
"""
COMPREHENSIVE ablation studies
This is what separates good papers from great ones
"""

from typing import Dict, List
import torch
from torch.utils.data import DataLoader, random_split
from models.tst_v2 import TemporalSpikingTransformer, LearnableTSA
from training.trainer_v2 import AdvancedTrainer
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import layer, functional


class TSAFixedDecay(LearnableTSA):
    """TSA with all neuron parameters frozen (tau, threshold, temperature)."""
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__(dim, num_heads, **kwargs)
        self.attn_tau = nn.Parameter(torch.zeros(num_heads), requires_grad=False)
        self.attn_threshold = nn.Parameter(torch.ones(num_heads), requires_grad=False)
        self.temperature = nn.Parameter(torch.ones(num_heads), requires_grad=False)


class StandardSpikeAttention(LearnableTSA):
    """Standard dot-product spiking attention — no membrane dynamics."""
    def compute_spike_attention(self, q, k, v, T):
        _, B, H, N, C = q.shape
        output = torch.zeros(T, B, N, H * C, device=q.device)
        total_spikes = 0
        spike_rate_per_timestep = []
        for t in range(T):
            # Direct softmax (no membrane integration)
            attn = torch.einsum('bhqc,bhkc->bhqk', q[t], k[t]) * (C ** -0.5)
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('bhqk,bhkc->bhqc', attn, v[t])
            output[t] = out.reshape(B, N, H * C)
            spikes = (q[t].abs() > 0).float().sum().item()
            total_spikes += spikes
            spike_rate_per_timestep.append(spikes / (B * H * N * C))
        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / max(T * B * H * N * C, 1),
            'spike_rate_per_timestep': spike_rate_per_timestep,
            'learned_beta': [0.0] * H,
            'learned_theta': [1.0] * H,
            'learned_temp': [1.0] * H,
        }
        return output, metrics


class NoAttention(nn.Module):
    """Identity attention — ablates the entire TSA block."""
    def __init__(self, dim, num_heads=8, **kwargs):
        super().__init__()
        self.proj = layer.Linear(dim, dim)
        functional.set_step_mode(self, 'm')

    def forward(self, x):
        # x: [T, B, N, D] — just project, no attention
        out = self.proj(x)
        metrics = {
            'attention': {
                'total_spikes': 0,
                'avg_spike_rate': 0.0,
                'spike_rate_per_timestep': [],
                'learned_beta': [],
                'learned_theta': [],
                'learned_temp': [],
            }
        }
        return out, metrics


def build_model_with_attention(attn_class, base_config: dict) -> nn.Module:
    """Build TemporalSpikingTransformer but patch its attention class."""
    from models.tst_v2 import TSABlock

    model = TemporalSpikingTransformer(**base_config)
    # Replace each block's attention with the ablation variant
    embed_dim = base_config.get('embed_dim', 256)
    num_heads = base_config.get('num_heads', 8)
    for blk in model.blocks:
        blk.attn = attn_class(dim=embed_dim, num_heads=num_heads)
    functional.set_step_mode(model, 'm')
    return model


def get_loaders_with_time_bins(n_bins: int, dataset: str = 'nmnist', batch_size: int = 32):
    """Create DataLoaders with a specific number of time bins."""

    if dataset == 'nmnist':
        sensor_size = (34, 34, 2)
        transform = transforms.Compose([
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, n_time_bins=n_bins),
        ])
        train_ds = datasets.NMNIST(save_to='./data', train=True, transform=transform)
        test_ds  = datasets.NMNIST(save_to='./data', train=False, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    train_size = int(0.9 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [train_size, len(train_ds) - train_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader, test_loader


class AblationFramework:
    """
    Systematic ablation study framework
    """
    
    def __init__(self, base_config: dict, dataset_loaders: tuple, dataset: str = 'nmnist'):
        self.base_config = base_config
        self.train_loader, self.val_loader, self.test_loader = dataset_loaders
        self.dataset = dataset
        self.results = {}
    
    def ablate_depth(self) -> Dict:
        """
        Ablation 1: Number of transformer blocks
        """
        depths = [2, 4, 6, 8, 12]
        results = {}
        
        for depth in depths:
            print(f"\n Testing depth={depth}")
            
            config = {**self.base_config, 'depth': depth}
            model = TemporalSpikingTransformer(**config)
            
            trainer = AdvancedTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                config={'epochs': 100, 'lr': 1e-3},  # Shorter for ablation
            )
            
            metrics = trainer.train()
            results[f'depth_{depth}'] = metrics['test_metrics']
        
        self.results['depth_ablation'] = results
        return results
    
    def ablate_num_heads(self) -> Dict:
        """
        Ablation 2: Number of attention heads
        """
        num_heads_list = [4, 8, 12, 16]
        results = {}
        
        for num_heads in num_heads_list:
            print(f"\nTesting num_heads={num_heads}")
            
            config = {**self.base_config, 'num_heads': num_heads}
            model = TemporalSpikingTransformer(**config)
            
            trainer = AdvancedTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                config={'epochs': 100, 'lr': 1e-3},
            )
            
            metrics = trainer.train()
            results[f'heads_{num_heads}'] = metrics['test_metrics']
        
        self.results['heads_ablation'] = results
        return results
    
    def ablate_attention_mechanism(self) -> Dict:
        """
        Ablation 3: Different attention mechanisms
        CRITICAL: Shows your TSA is necessary
        """
        attention_types = {
            'TSA (ours)': LearnableTSA,
            'TSA_fixed_decay': TSAFixedDecay,
            'Standard_spike_attn': StandardSpikeAttention,
            'No_attention': NoAttention,
        }
        
        results = {}
        
        for name, attn_class in attention_types.items():
            print(f"\nTesting {name}")
            
            # Build model with this attention type
            model = build_model_with_attention(attn_class, self.base_config)
            
            trainer = AdvancedTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                config={'epochs': 100, 'lr': 1e-3},
            )
            
            metrics = trainer.train()
            results[name] = metrics['test_metrics']
        
        self.results['attention_ablation'] = results
        return results
    
    def ablate_spike_regularization(self) -> Dict:
        """
        Ablation 4: Energy regularization weight
        """
        spike_reg_weights = [0.0, 0.0001, 0.001, 0.01, 0.1]
        results = {}
        
        for spike_reg in spike_reg_weights:
            print(f"\nTesting spike_reg={spike_reg}")
            
            model = TemporalSpikingTransformer(**self.base_config)
            
            trainer = AdvancedTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                config={
                    'epochs': 100,
                    'lr': 1e-3,
                    'spike_reg': spike_reg,
                },
            )
            
            metrics = trainer.train()
            results[f'reg_{spike_reg}'] = metrics['test_metrics']
        
        self.results['regularization_ablation'] = results
        return results
    
    def ablate_time_bins(self) -> Dict:
        """
        Ablation 5: Number of time bins
        """
        time_bins_list = [10, 20, 40, 80]
        results = {}
        
        for n_bins in time_bins_list:
            print(f"\nTesting time_bins={n_bins}")
            
            # Recreate data loaders with new time bins
            train_loader, val_loader, test_loader = get_loaders_with_time_bins(n_bins, dataset=self.dataset)
            
            model = TemporalSpikingTransformer(**self.base_config)
            
            trainer = AdvancedTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config={'epochs': 100, 'lr': 1e-3},
            )
            
            metrics = trainer.train()
            results[f'bins_{n_bins}'] = metrics['test_metrics']
        
        self.results['time_bins_ablation'] = results
        return results
    
    def ablate_neuron_parameters(self) -> Dict:
        """
        Ablation 6: Fixed vs learnable neuron parameters
        """
        configs = {
            'learnable_all': {'learnable_tau': True, 'learnable_threshold': True},
            'learnable_tau_only': {'learnable_tau': True, 'learnable_threshold': False},
            'learnable_threshold_only': {'learnable_tau': False, 'learnable_threshold': True},
            'fixed_all': {'learnable_tau': False, 'learnable_threshold': False},
        }
        
        results = {}
        
        for name, config in configs.items():
            print(f"\nTesting {name}")
            
            model_config = {**self.base_config, **config}
            model = TemporalSpikingTransformer(**model_config)
            
            trainer = AdvancedTrainer(
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                config={'epochs': 100, 'lr': 1e-3},
            )
            
            metrics = trainer.train()
            results[name] = metrics['test_metrics']
        
        self.results['neuron_params_ablation'] = results
        return results
    
    def run_all_ablations(self) -> Dict:
        """
        Run complete ablation study
        This takes TIME but is ESSENTIAL
        """
        print("\nRunning comprehensive ablation studies...")
        
        self.ablate_depth()
        self.ablate_num_heads()
        self.ablate_attention_mechanism()
        self.ablate_spike_regularization()
        self.ablate_time_bins()
        self.ablate_neuron_parameters()
        
        # Save all results
        self.save_ablation_results()
        
        # Generate plots
        self.generate_ablation_plots()
        
        return self.results
    
    def save_ablation_results(self):
        """Save to JSON"""
        import json
        with open('ablation_results.json', 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_ablation_plots(self):
        """
        Generate publication-quality plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        sns.set_style('whitegrid')
        sns.set_palette('husl')
        
        # Plot 1: Depth ablation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        depths = [2, 4, 6, 8, 12]
        accs = [self.results['depth_ablation'][f'depth_{d}']['acc'] for d in depths]
        energies = [self.results['depth_ablation'][f'depth_{d}']['avg_energy_uJ'] for d in depths]
        
        ax1.plot(depths, accs, marker='o', linewidth=2, markersize=8)
        ax1.set_xlabel('Number of Blocks', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Effect of Model Depth', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(depths, energies, marker='s', linewidth=2, markersize=8, color='orange')
        ax2.set_xlabel('Number of Blocks', fontsize=12)
        ax2.set_ylabel('Energy (μJ)', fontsize=12)
        ax2.set_title('Energy vs Depth', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ablation_depth.pdf', dpi=300, bbox_inches='tight')
        
        # Plot 2: Attention mechanism comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        
        attn_names = list(self.results['attention_ablation'].keys())
        attn_accs = [self.results['attention_ablation'][name]['acc'] * 100 
                     for name in attn_names]
        attn_energies = [self.results['attention_ablation'][name]['avg_energy_uJ'] 
                        for name in attn_names]
        
        scatter = ax.scatter(attn_energies, attn_accs, s=200, alpha=0.7)
        
        for i, name in enumerate(attn_names):
            ax.annotate(name, (attn_energies[i], attn_accs[i]), 
                       fontsize=11, ha='center', va='bottom')
        
        ax.set_xlabel('Energy (μJ)', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Attention Mechanism Comparison', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.savefig('ablation_attention.pdf', dpi=300, bbox_inches='tight')
        
        print("Ablation plots saved!")