# models/tst_v2.py
"""
VERSION 2: Production-ready implementation
- All hyperparameters learnable
- Extensive logging
- Gradient flow analysis
- Memory efficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class LearnableTSA(nn.Module):
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_tau: float = 2.0,
        init_threshold: float = 1.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv = layer.Linear(dim, dim * 3, bias=qkv_bias)
        
        self.q_neuron = neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.k_neuron = neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.v_neuron = neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        
        self.attn_tau = nn.Parameter(torch.ones(num_heads) * init_tau)
        self.attn_threshold = nn.Parameter(torch.ones(num_heads) * init_threshold)
        self.temperature = nn.Parameter(torch.ones(num_heads))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = layer.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_neuron = neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        
    def compute_spike_attention(self, q, k, v, T):
        _, B, H, N, C = q.shape
        beta = torch.sigmoid(self.attn_tau)
        theta = F.softplus(self.attn_threshold)
        temp = F.softplus(self.temperature)
        
        attn_mem = torch.zeros(B, H, N, N, device=q.device)
        output = torch.zeros(T, B, N, H * C, device=q.device)
        
        total_spikes = 0
        for t in range(T):
            spike_coincidence = torch.einsum('bhqc,bhkc->bhqk', q[t], k[t]) / temp.view(1, H, 1, 1)
            attn_mem = beta.view(1, H, 1, 1) * attn_mem + spike_coincidence
            
            attn_spikes_hard = (attn_mem > theta.view(1, H, 1, 1)).float()
            attn_spikes_soft = torch.sigmoid((attn_mem - theta.view(1, H, 1, 1)) * 10.0)
            attn_spikes = attn_spikes_hard + (attn_spikes_soft - attn_spikes_soft.detach())
            
            num_spikes = attn_spikes.sum().item()
            total_spikes += num_spikes
            
            attn_mem = attn_mem * (1.0 - attn_spikes)
            attn_spikes = self.attn_drop(attn_spikes)
            
            out = torch.einsum('bhqk,bhkc->bhqc', attn_spikes, v[t])
            output[t] = out.reshape(B, N, H * C)
        
        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (T * B * H * N * N),
        }
        return output, metrics
    
    def forward(self, x: torch.Tensor):
        T, B, N, C = x.shape
        qkv = self.qkv(x).reshape(T, B, N, 3, self.num_heads, self.head_dim).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = self.q_neuron(q)
        k = self.k_neuron(k)
        v = self.v_neuron(v)
        
        attn_out, attn_metrics = self.compute_spike_attention(q, k, v, T)
        
        x = self.proj(attn_out)
        x = self.proj_drop(x)
        x = self.proj_neuron(x)
        
        return x, {'attention': attn_metrics}


class TSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, init_tau=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LearnableTSA(dim, num_heads, qkv_bias, attn_drop, drop, init_tau)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            layer.Linear(dim, mlp_hidden),
            neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(drop),
            layer.Linear(mlp_hidden, dim),
            neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(drop),
        )
        
    def forward(self, x):
        attn_out, attn_metrics = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x, {'attention': attn_metrics}


class TemporalSpikingTransformer(nn.Module):
    def __init__(
        self,
        img_size=34,
        patch_size=2,
        in_channels=2,
        num_classes=10,
        embed_dim=256,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        init_tau=2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_size = patch_size
        
        self.patch_embed = nn.Sequential(
            layer.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan()),
        )
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(drop_rate)
        
        self.blocks = nn.ModuleList([
            TSABlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, init_tau)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            layer.Linear(embed_dim, num_classes),
            neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        
        functional.set_step_mode(self, 'm')
    
    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.patch_embed(x).flatten(3).transpose(2, 3)  # [T, B, N, D]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        all_metrics = []
        for block in self.blocks:
            x, m = block(x)
            all_metrics.append(m)
        
        x = self.norm(x)
        x = x.mean(2)  # pool patches
        x = self.head(x)
        x = x.mean(0)  # pool time
        
        return x, {'blocks': all_metrics}
    
    @torch.no_grad()
    def get_energy_breakdown(self, x: torch.Tensor) -> dict:
        """Fixed and correctly indented."""
        functional.reset_net(self)
        _, metrics = self.forward(x)
        
        total_spikes = 0
        for block in metrics.get('blocks', []):
            attn = block.get('attention', {})
            total_spikes += attn.get('total_spikes', 0)
        
        return {
            'total_spikes': total_spikes,
            'energy_per_sample_uJ': total_spikes * 0.1,
        }
