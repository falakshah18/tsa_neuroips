# models/tst_v2.py
"""
Temporal Spiking Attention (TSA) and the Temporal Spiking Transformer (TST).

This module implements the paper's core architectural contribution: a
spiking-neuron-based replacement for softmax attention.

Standard transformer attention computes a continuous weight for every
query/key pair via softmax(QK^T / sqrt(d)). TSA replaces this with genuine
spiking-neuron dynamics. For each head, a leaky "attention membrane"
integrates query/key coincidence over time:

    m_t = beta * m_{t-1} + (q_t . k_t) / temperature

where beta = sigmoid(attn_tau) is a per-head, *learnable* decay rate (as
opposed to a fixed hyperparameter). A pair only "attends" to each other at
timestep t if the membrane crosses a per-head, learnable threshold
theta = softplus(attn_threshold):

    spike_t = 1[m_t > theta]          (hard, used in the forward pass)
    spike_t ~= sigmoid(10 * (m_t - theta))   (soft, used for the backward pass)

The soft/hard split (see `compute_spike_attention`) is a straight-through
surrogate-gradient estimator: the forward pass uses the true binary spike
(so the model's actual behavior is genuinely event-driven / sparse), while
gradients flow through the smooth sigmoid approximation, since the hard
step function has zero gradient almost everywhere.

Class hierarchy:
    LearnableTSA                 -- one spiking multi-head attention layer
    TSABlock                     -- LearnableTSA + spiking MLP + residuals
    TemporalSpikingTransformer   -- patch embedding + N x TSABlock + head

All modules operate in SpikingJelly's multi-step mode: every tensor carries
an explicit leading time dimension T, i.e. shape [T, B, ...], and neuron
state (membrane potentials) persists across calls until
`spikingjelly.activation_based.functional.reset_net()` is invoked --
typically once per training/eval batch, since each batch is a fresh sample
sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, surrogate, layer, functional
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class LearnableTSA(nn.Module):
    """
    Multi-head Temporal Spiking Attention.

    Drop-in spiking replacement for standard multi-head softmax attention.
    Queries, keys, and values are each passed through their own spiking
    (LIF) neuron before the attention computation itself, so by the time
    `compute_spike_attention` runs, q/k/v are already binary spike trains
    rather than continuous activations.

    Args:
        dim: Total embedding dimension (must be divisible by num_heads).
        num_heads: Number of attention heads.
        qkv_bias: Whether the Q/K/V projection includes a bias term.
        attn_drop: Dropout probability applied to attention spikes.
        proj_drop: Dropout probability applied after the output projection.
        init_tau: Initial value for each neuron's membrane time constant.
        init_threshold: Initial value for the per-head attention threshold
            (before the softplus that keeps it positive).
        learnable_tau: If True, the per-head decay rate `attn_tau` is a
            trainable parameter; if False, it stays fixed at its init value.
        learnable_threshold: Same, but for `attn_threshold`.

    Shape:
        Input: [T, B, N, dim] (T timesteps, batch B, N tokens/patches)
        Output: [T, B, N, dim], plus a metrics dict with spike statistics.
    """

    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        init_tau: float = 2.0,
        init_threshold: float = 1.0,
        learnable_tau: bool = True,
        learnable_threshold: bool = True,
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
        
        self.attn_tau = nn.Parameter(torch.ones(num_heads) * init_tau, requires_grad=learnable_tau)
        self.attn_threshold = nn.Parameter(torch.ones(num_heads) * init_threshold, requires_grad=learnable_threshold)
        self.temperature = nn.Parameter(torch.ones(num_heads))
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = layer.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_neuron = neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True)
        
    def compute_spike_attention(self, q, k, v, T):
        """
        Run the leaky-integrate-and-fire attention dynamics over T timesteps.

        For each head h, maintains a membrane potential `attn_mem` per
        query/key pair, updated as:
            attn_mem = beta_h * attn_mem + (q_t . k_t) / temperature_h
        A spike fires (and the membrane resets toward 0) wherever
        `attn_mem` exceeds the per-head threshold theta_h. Spiking pairs
        gate how much of v_t is read into the output at that timestep,
        exactly as attention weights would in standard softmax attention --
        except the "weight" here is strictly binary (0 or 1) per pair per
        step, not a continuous value.

        Args:
            q, k, v: Spike tensors of shape [T, B, num_heads, N, head_dim],
                already passed through their respective LIF neurons.
            T: Number of timesteps (must match q/k/v's leading dimension).

        Returns:
            output: Tensor of shape [T, B, N, num_heads * head_dim].
            metrics: Dict with 'total_spikes' (int) and 'avg_spike_rate'
                (fraction of all query/key/timestep/head cells that spiked)
                -- used for the energy regularizer and reporting.
        """
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
            output[t] = out.permute(0, 2, 1, 3).reshape(B, N, H * C)
        
        metrics = {
            'total_spikes': total_spikes,
            'avg_spike_rate': total_spikes / (T * B * H * N * N),
        }
        return output, metrics
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [T, B, N, dim] input sequence.

        Returns:
            Tuple of (output [T, B, N, dim], metrics dict with key
            'attention' containing spike statistics from
            `compute_spike_attention`).
        """
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
    """
    One transformer block: LearnableTSA attention + spiking MLP, each with
    a residual connection and LayerNorm (post-norm, following the original
    Transformer rather than the pre-norm variant).

    Args:
        dim: Embedding dimension.
        num_heads: Number of attention heads (passed to LearnableTSA).
        mlp_ratio: Hidden-layer width of the MLP as a multiple of `dim`.
        qkv_bias, drop, attn_drop, init_tau, learnable_tau,
        learnable_threshold: Passed through to LearnableTSA; see its
            docstring for details.
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, init_tau=2.0,
                 learnable_tau=True, learnable_threshold=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LearnableTSA(dim, num_heads, qkv_bias, attn_drop, drop, init_tau,
                                  learnable_tau=learnable_tau, learnable_threshold=learnable_threshold)
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
        """
        Args:
            x: [T, B, N, dim] input sequence.

        Returns:
            Tuple of (output [T, B, N, dim], metrics dict passed through
            unchanged from LearnableTSA.forward).
        """
        attn_out, attn_metrics = self.attn(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mlp(x))
        return x, {'attention': attn_metrics}


class TemporalSpikingTransformer(nn.Module):
    """
    Full TSA-based vision transformer for event-based (neuromorphic) data.

    Pipeline: Conv2d patch embedding (with its own LIF neuron) -> learnable
    positional embedding -> `depth` x TSABlock -> LayerNorm -> mean-pool
    over patches -> spiking classification head -> mean-pool over time.

    Architecturally this is a standard Vision Transformer with every dense
    attention/MLP activation replaced by spiking-neuron dynamics, and
    softmax attention replaced by LearnableTSA. The patch embedding is a
    genuine 2D convolution, so this model expects real 2D spatial
    structure in its input (e.g. N-MNIST's 34x34 event frames) -- it is
    not currently suited to 1D spike trains like SHD's 700-channel audio
    data, which has no such spatial structure to patchify.

    Args:
        img_size: Height/width of the (square) input frames.
        patch_size: Side length of each square patch; num_patches =
            (img_size // patch_size) ** 2.
        in_channels: Number of input channels (e.g. 2 for DVS on/off
            polarity channels).
        num_classes: Number of output classes.
        embed_dim: Transformer embedding dimension.
        depth: Number of TSABlocks stacked.
        num_heads: Attention heads per block.
        mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, init_tau,
        learnable_tau, learnable_threshold: Passed through to each
            TSABlock; see LearnableTSA's docstring for details.
    """

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
        learnable_tau=True,
        learnable_threshold=True,
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
            TSABlock(embed_dim, num_heads, mlp_ratio, qkv_bias, drop_rate, attn_drop_rate, init_tau,
                     learnable_tau=learnable_tau, learnable_threshold=learnable_threshold)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            layer.Linear(embed_dim, num_classes),
            neuron.ParametricLIFNode(init_tau=init_tau, surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        
        functional.set_step_mode(self, 'm')
    
    def forward(self, x):
        """
        Args:
            x: [T, B, C, H, W] input event frames (T timesteps).

        Returns:
            Tuple of:
                logits: [B, num_classes], averaged over the time dimension
                    (i.e. the model votes at every timestep and the final
                    prediction is the mean of those votes).
                metrics: Dict with key 'blocks', a list of one metrics
                    dict per TSABlock (see LearnableTSA.forward), used for
                    energy estimation and spike-rate regularization.
        """
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
        """
        Estimate inference energy from total spike count.

        Runs a single forward pass (resetting neuron state first) and
        converts the total number of spikes across all attention layers
        into an energy estimate, using a fixed per-spike energy cost. This
        is the standard first-order neuromorphic energy proxy: each spike
        corresponds to one synaptic event, and 0.1 pJ/spike is a
        commonly-cited figure for digital neuromorphic hardware (e.g.
        Loihi-class chips) -- see `hardware/loihi2_deployment.py` for a
        more detailed, hardware-specific estimate.

        Args:
            x: [T, B, C, H, W] input batch to run inference on.

        Returns:
            Dict with:
                total_spikes: Total spike count across all attention layers.
                total_energy_J: total_spikes * 0.1e-12 (Joules).
                energy_per_sample_uJ: total_energy_J normalized by batch
                    size, in microjoules.
        """
        functional.reset_net(self)
        _, metrics = self.forward(x)
        
        total_spikes = 0
        for block in metrics.get('blocks', []):
            attn = block.get('attention', {})
            total_spikes += attn.get('total_spikes', 0)
        
        # Energy constant: 0.1 pJ per spike (consistent with training/trainer_v2.py).
        energy_per_spike_j = 0.1e-12
        total_energy_j = total_spikes * energy_per_spike_j
        batch_size = x.shape[1]

        return {
            'total_spikes': total_spikes,
            'total_energy_J': total_energy_j,
            'energy_per_sample_uJ': (total_energy_j / max(batch_size, 1)) * 1e6,
        }
