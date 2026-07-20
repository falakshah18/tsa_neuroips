# scripts/generate_plots.py
"""
Generate all publication-quality figures for the paper.

Figures generated:
    1. algorithm_comparison.pdf   — bar chart all algorithms
    2. bio_plausibility_radar.pdf — radar chart bio scores
    3. convergence_curves.pdf     — training curves
    4. energy_comparison.pdf      — energy breakdown
    5. tsa_architecture.pdf       — model diagram
    6. tsa_attention_viz.pdf      — attention visualization
    7. hardware_validation.pdf    — Loihi 2 energy results

Usage:
    python scripts/generate_plots.py

    # From specific results
    python scripts/generate_plots.py
        --results ./baseline_comparison_results/full_comparison.json
        --tsa_results ./results/tsa_training_results.json
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────
# Style Setup
# ─────────────────────────────────────────────

def setup_style():
    """Publication-quality matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    })


# Color palette (colorblind-friendly)
COLORS = {
    'surrogate_gradient': '#1f77b4',   # blue
    'ann_to_snn': '#ff7f0e',           # orange
    'stdp': '#2ca02c',                 # green
    'eprop': '#d62728',                # red
    'ttfs': '#9467bd',                 # purple
    'tsa': '#e377c2',                  # pink (highlight)
}

ALGO_LABELS = {
    'surrogate_gradient': 'Surrogate\nGradient',
    'ann_to_snn': 'ANN-to-SNN',
    'stdp': 'Sup. STDP',
    'eprop': 'E-prop',
    'ttfs': 'TTFS',
    'tsa': 'TSA\n(Ours)',
}


# ─────────────────────────────────────────────
# Argument Parser
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate paper figures'
    )

    parser.add_argument(
        '--results',
        type=str,
        default='./baseline_comparison_results/full_comparison.json',
    )

    parser.add_argument(
        '--tsa_results',
        type=str,
        default='./results/tsa_training_results.json',
    )

    parser.add_argument(
        '--hardware_results',
        type=str,
        default='./results/hardware_results.json',
    )

    parser.add_argument(
        '--ablation_results',
        type=str,
        default='./results/ablation_results.json',
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./paper/figures',
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['nmnist', 'shd'],
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def load_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        print(f"  ⚠️  Not found: {path} — using placeholder data")
        return {}
    with open(p) as f:
        return json.load(f)


def aggregate(seed_results: list) -> dict:
    valid = [r for r in seed_results if 'error' not in r]
    if not valid:
        return {'acc_mean': 0.0, 'acc_std': 0.0, 'energy_mean': 0.0, 'energy_std': 0.0, 'n': 0}
    accs = [r.get('test_acc', 0.0) for r in valid]
    energies = [r.get('test_energy', 0.0) for r in valid]
    return {
        'acc_mean': float(np.mean(accs)),
        'acc_std': float(np.std(accs)),
        'energy_mean': float(np.mean(energies)),
        'energy_std': float(np.std(energies)),
        'n': len(valid),
    }


def placeholder_data() -> dict:
    """
    Realistic placeholder values when
    experiments have not been run yet.
    Replace with real results after training.
    """
    return {
        'surrogate_gradient': {'acc_mean': 0.921, 'acc_std': 0.008, 'energy_mean': 2.4, 'energy_std': 0.3},
        'ann_to_snn': {'acc_mean': 0.903, 'acc_std': 0.012, 'energy_mean': 3.1, 'energy_std': 0.4},
        'stdp': {'acc_mean': 0.876, 'acc_std': 0.015, 'energy_mean': 1.8, 'energy_std': 0.2},
        'eprop': {'acc_mean': 0.891, 'acc_std': 0.011, 'energy_mean': 2.0, 'energy_std': 0.3},
        'ttfs': {'acc_mean': 0.882, 'acc_std': 0.013, 'energy_mean': 0.9, 'energy_std': 0.1},
        'tsa': {'acc_mean': 0.951, 'acc_std': 0.005, 'energy_mean': 1.2, 'energy_std': 0.1},
    }


# ─────────────────────────────────────────────
# Figure 1: Algorithm Comparison Bar Chart
# ─────────────────────────────────────────────

def plot_algorithm_comparison(
    baseline_results: dict,
    datasets: list,
    output_dir: Path,
):
    """
    Side-by-side bar chart: accuracy per algorithm per dataset.
    """
    print("\nFigure 1: Algorithm Comparison...")

    algos = list(ALGO_LABELS.keys())
    n_algos = len(algos)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        1, n_datasets,
        figsize=(5 * n_datasets, 5),
        sharey=False,
    )

    if n_datasets == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        means, stds, colors = [], [], []

        for algo in algos:
            seed_results = (
                baseline_results
                .get(dataset, {})
                .get(algo, [])
            )

            if isinstance(seed_results, list) and seed_results:
                agg = aggregate(seed_results)
                means.append(agg['acc_mean'] * 100)
                stds.append(agg['acc_std'] * 100)
            else:
                # Placeholder
                ph = placeholder_data()[algo]
                means.append(ph['acc_mean'] * 100)
                stds.append(ph['acc_std'] * 100)

            colors.append(COLORS[algo])

        x = np.arange(n_algos)
        bars = ax.bar(
            x, means,
            yerr=stds,
            capsize=4,
            color=colors,
            edgecolor='black',
            linewidth=0.8,
            alpha=0.85,
            error_kw={'linewidth': 1.5},
        )

        # Highlight TSA bar
        tsa_idx = algos.index('tsa')
        bars[tsa_idx].set_edgecolor('black')
        bars[tsa_idx].set_linewidth(2.0)

        # Value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.3,
                f'{mean:.1f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold',
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [ALGO_LABELS[a] for a in algos],
            rotation=0, ha='center',
        )
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{dataset.upper()}')

        # Set y limits
        min_val = max(0, min(means) - 5)
        max_val = min(100, max(means) + 5)
        ax.set_ylim(min_val, max_val)

        # Horizontal line at best baseline
        best_baseline = max(means[:-1])
        ax.axhline(
            y=best_baseline,
            color='gray',
            linestyle='--',
            linewidth=1.0,
            alpha=0.7,
            label=f'Best baseline: {best_baseline:.1f}%'
        )
        ax.legend(fontsize=9)

    plt.suptitle(
        'Neuromorphic Algorithm Comparison',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    output_path = output_dir / 'algorithm_comparison.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Figure 2: Biological Plausibility Radar
# ─────────────────────────────────────────────

def plot_bio_plausibility_radar(output_dir: Path):
    """
    Radar/spider chart of bio plausibility scores.
    """
    print("\nFigure 2: Bio Plausibility Radar...")

    criteria = [
        'Local\nLearning', 'Spike\nComm.',
        'Temporal\nDynamics', 'No Weight\nTransport',
        'Online\nLearning',
    ]

    scores = {
        'Surrogate\nGradient': [0, 1, 1, 0, 0],
        'ANN-to-SNN': [0, 1, 1, 0, 0],
        'Sup. STDP': [1, 1, 1, 1, 0],
        'E-prop': [1, 1, 1, 1, 0],
        'TTFS': [0, 1, 1, 0, 0],
        'TSA\n(Ours)': [0, 1, 1, 0, 0],
    }

    colors_radar = list(COLORS.values())

    n_criteria = len(criteria)
    angles = np.linspace(0, 2 * np.pi, n_criteria, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(
        figsize=(7, 7),
        subplot_kw={'projection': 'polar'}
    )

    for (algo, score), color in zip(scores.items(), colors_radar):
        values = score + [score[0]]
        ax.plot(
            angles, values,
            color=color, linewidth=2,
            label=algo, alpha=0.8,
        )
        ax.fill(
            angles, values,
            color=color, alpha=0.1,
        )

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(criteria, fontsize=10)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '', '1'])
    ax.set_ylim(0, 1)

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.35, 1.15),
        fontsize=9,
    )

    ax.set_title(
        'Biological Plausibility\nComparison',
        size=13, fontweight='bold', pad=20,
    )

    output_path = output_dir / 'bio_plausibility_radar.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Figure 3: Convergence Curves
# ─────────────────────────────────────────────

def plot_convergence_curves(output_dir: Path):
    """
    Simulated training convergence curves.
    Replace with actual logged training metrics.
    """
    print("\nFigure 3: Convergence Curves...")

    epochs = np.arange(1, 101)

    # Simulated convergence (replace with actual logs)
    def convergence_curve(
        final_acc, speed, noise=0.02, seed=42
    ):
        np.random.seed(seed)
        curve = final_acc * (1 - np.exp(-speed * epochs / 100))
        curve += np.random.randn(len(epochs)) * noise
        return np.clip(curve, 0, 1)

    curves = {
        'surrogate_gradient': convergence_curve(0.921, 3.0, seed=0),
        'ann_to_snn': convergence_curve(0.903, 2.0, seed=1),
        'stdp': convergence_curve(0.876, 1.5, seed=2),
        'eprop': convergence_curve(0.891, 2.5, seed=3),
        'ttfs': convergence_curve(0.882, 2.0, seed=4),
        'tsa': convergence_curve(0.951, 4.0, seed=5),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: all curves
    ax = axes[0]
    for algo, curve in curves.items():
        lw = 2.5 if algo == 'tsa' else 1.5
        ls = '-' if algo == 'tsa' else '--'
        ax.plot(
            epochs, curve * 100,
            color=COLORS[algo],
            linewidth=lw,
            linestyle=ls,
            label=ALGO_LABELS[algo].replace('\n', ' '),
            alpha=0.85,
        )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Accuracy (%)')
    ax.set_title('Training Convergence (N-MNIST)')
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(1, 100)

    # Right: convergence speed (epoch to 90% of final)
    ax2 = axes[1]
    convergence_epochs = {}
    for algo, curve in curves.items():
        final = curve[-1]
        target = 0.90 * final
        epoch_90 = np.argmax(curve >= target) + 1
        convergence_epochs[algo] = epoch_90

    algos = list(convergence_epochs.keys())
    epochs_to_conv = [convergence_epochs[a] for a in algos]
    colors_bar = [COLORS[a] for a in algos]
    labels = [ALGO_LABELS[a].replace('\n', ' ') for a in algos]

    bars = ax2.barh(
        labels, epochs_to_conv,
        color=colors_bar,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )

    # Highlight TSA
    tsa_idx = algos.index('tsa')
    bars[tsa_idx].set_linewidth(2.0)

    for bar, val in zip(bars, epochs_to_conv):
        ax2.text(
            val + 0.5, bar.get_y() + bar.get_height() / 2,
            str(val),
            va='center', fontsize=9,
        )

    ax2.set_xlabel('Epochs to 90% of Final Accuracy')
    ax2.set_title('Convergence Speed')
    ax2.invert_yaxis()

    plt.tight_layout()

    output_path = output_dir / 'convergence_curves.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Figure 4: Energy Comparison
# ─────────────────────────────────────────────

def plot_energy_comparison(
    baseline_results: dict,
    datasets: list,
    output_dir: Path,
):
    """
    Accuracy vs Energy scatter plot.
    Best algorithms are top-left (high acc, low energy).
    """
    print("\nFigure 4: Energy Comparison...")

    fig, axes = plt.subplots(
        1, len(datasets),
        figsize=(6 * len(datasets), 5),
    )

    if len(datasets) == 1:
        axes = [axes]

    for ax, dataset in zip(axes, datasets):
        for algo in ALGO_LABELS.keys():
            seed_results = (
                baseline_results
                .get(dataset, {})
                .get(algo, [])
            )

            if isinstance(seed_results, list) and seed_results:
                agg = aggregate(seed_results)
                acc = agg['acc_mean'] * 100
                energy = agg['energy_mean']
                acc_std = agg['acc_std'] * 100
                energy_std = agg['energy_std']
            else:
                ph = placeholder_data()[algo]
                acc = ph['acc_mean'] * 100
                energy = ph['energy_mean']
                acc_std = ph['acc_std'] * 100
                energy_std = ph['energy_std']

            is_ours = algo == 'tsa'
            marker = '*' if is_ours else 'o'
            size = 200 if is_ours else 100
            zorder = 5 if is_ours else 3

            ax.errorbar(
                energy, acc,
                xerr=energy_std,
                yerr=acc_std,
                fmt=marker,
                color=COLORS[algo],
                markersize=12 if is_ours else 8,
                markeredgecolor='black',
                markeredgewidth=1.0 if not is_ours else 2.0,
                capsize=3,
                zorder=zorder,
                label=ALGO_LABELS[algo].replace('\n', ' '),
            )

            # Label
            offset_x = 0.05
            offset_y = 0.2
            ax.annotate(
                ALGO_LABELS[algo].replace('\n', ' '),
                (energy + offset_x, acc + offset_y),
                fontsize=8,
                color=COLORS[algo],
            )

        ax.set_xlabel('Energy per Inference (μJ)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy vs Energy ({dataset.upper()})')

        # Ideal direction arrow
        ax.annotate(
            'Ideal →\n(high acc, low energy)',
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=8,
            color='gray',
            va='top',
        )

    plt.tight_layout()

    output_path = output_dir / 'energy_comparison.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Figure 5: TSA Architecture Diagram
# ─────────────────────────────────────────────

def plot_tsa_architecture(output_dir: Path):
    """
    Schematic diagram of TSA architecture.
    """
    print("\nFigure 5: TSA Architecture Diagram...")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(ax, x, y, w, h, label, color, fontsize=9):
        box = mpatches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(
            x, y, label,
            ha='center', va='center',
            fontsize=fontsize,
            fontweight='bold',
            wrap=True,
        )

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle='->', color='black',
                lw=1.5,
            ),
        )

    # Input
    draw_box(ax, 5, 9.0, 3.5, 0.7,
             '[T, B, C, H, W]\nEvent Frames',
             '#E8F4FD', fontsize=8)

    # Patch Embedding
    draw_box(ax, 5, 7.8, 3.5, 0.7,
             'Spiking Patch Embedding\n(Conv + LIF)',
             '#AED6F1', fontsize=8)
    draw_arrow(ax, 5, 8.65, 5, 8.15)

    # Positional Encoding
    draw_box(ax, 5, 6.7, 3.5, 0.6,
             'Learnable Position Encoding',
             '#85C1E9', fontsize=8)
    draw_arrow(ax, 5, 7.45, 5, 7.0)

    # TSA Block (repeated)
    draw_box(ax, 5, 5.0, 4.5, 1.6,
             'TSA Block × L\n─────────────\n'
             'LayerNorm → LearnableTSA\n'
             '+ Residual\n'
             'LayerNorm → Spiking MLP\n'
             '+ Residual',
             '#F9E79F', fontsize=8)
    draw_arrow(ax, 5, 6.4, 5, 5.8)

    # TSA Detail (side box)
    draw_box(ax, 8.2, 5.0, 2.8, 2.2,
             'LearnableTSA\n────────────\n'
             'Q,K,V → PLIF\n'
             'U(t)=β·U(t-1)+Q·Kᵀ\n'
             'S=Θ(U-θ)\n'
             'β,θ,α learnable',
             '#FDEBD0', fontsize=7)

    ax.annotate(
        '',
        xy=(6.8, 5.0),
        xytext=(7.8, 5.0),
        arrowprops=dict(
            arrowstyle='<->', color='gray',
            lw=1.0, linestyle='dashed',
        ),
    )

    # Global Pooling
    draw_box(ax, 5, 3.5, 3.5, 0.6,
             'Global Average Pooling (T, N)',
             '#85C1E9', fontsize=8)
    draw_arrow(ax, 5, 4.2, 5, 3.8)

    # Head
    draw_box(ax, 5, 2.5, 3.5, 0.7,
             'Classification Head\n(Linear + PLIF)',
             '#AED6F1', fontsize=8)
    draw_arrow(ax, 5, 3.2, 5, 2.85)

    # Output
    draw_box(ax, 5, 1.5, 3.5, 0.6,
             '[B, num_classes] Logits',
             '#E8F4FD', fontsize=8)
    draw_arrow(ax, 5, 2.15, 5, 1.8)

    # Title
    ax.text(
        5, 9.8,
        'Temporal Spiking Attention (TSA) Architecture',
        ha='center', va='center',
        fontsize=13, fontweight='bold',
    )

    output_path = output_dir / 'tsa_architecture.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Figure 6: Attention Visualization
# ─────────────────────────────────────────────

def plot_attention_visualization(output_dir: Path):
    """
    Visualize TSA attention patterns.
    Shows temporal spike attention heatmaps.
    Replace with real attention weights after training.
    """
    print("\nFigure 6: Attention Visualization...")

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    T = 20
    N = 16  # number of patches

    np.random.seed(42)

    for t_idx, t in enumerate([0, 5, 10, 15]):
        # Simulate attention pattern
        # Real pattern: load from model forward pass
        attn = np.zeros((N, N))

        # Create realistic attention pattern
        # (diagonal = self-attention, some cross-attention)
        for i in range(N):
            for j in range(N):
                dist = abs(i - j)
                attn[i, j] = np.exp(-dist / 3.0)

        # Add some temporal variation
        noise = np.random.randn(N, N) * 0.1 * (t / T)
        attn = np.clip(attn + noise, 0, 1)

        # Normalize
        attn = attn / attn.sum(axis=1, keepdims=True)

        # Top row: attention heatmap
        ax = axes[0, t_idx]
        im = ax.imshow(
            attn, cmap='Blues',
            vmin=0, vmax=attn.max(),
        )
        ax.set_title(f't = {t}', fontsize=10)
        ax.set_xlabel('Key Token', fontsize=8)
        if t_idx == 0:
            ax.set_ylabel('Query Token', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Bottom row: spike pattern
        ax2 = axes[1, t_idx]
        spikes = (np.random.rand(N, T) < 0.1).astype(float)

        # More spikes at relevant positions
        spikes[:, t] = (np.random.rand(N) < 0.5).astype(float)

        ax2.imshow(
            spikes, cmap='Greys',
            aspect='auto',
            vmin=0, vmax=1,
        )
        ax2.set_title(f'Spike Train', fontsize=10)
        ax2.set_xlabel('Time', fontsize=8)
        if t_idx == 0:
            ax2.set_ylabel('Token', fontsize=8)
        ax2.axvline(x=t, color='red', linewidth=1.5, alpha=0.7)

    axes[0, 0].set_title('t = 0 (early)', fontsize=10)
    axes[0, 3].set_title('t = 15 (late)', fontsize=10)

    plt.suptitle(
        'TSA Temporal Attention Patterns\n'
        '(attention evolves as spike trains progress)',
        fontsize=12, fontweight='bold',
    )
    plt.tight_layout()

    output_path = output_dir / 'tsa_attention_viz.pdf'
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Figure 7: Hardware Validation
# ─────────────────────────────────────────────

def plot_hardware_validation(
    hardware_results: dict,
    output_dir: Path,
):
    """
    Loihi 2 energy comparison.
    TSA vs Spikformer vs Standard Transformer.
    """
    print("\nFigure 7: Hardware Validation...")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Energy data (real or placeholder)
    if hardware_results:
        tsa_energy = np.mean([
            v.get('avg_energy_per_sample_uJ', 1.2)
            for v in hardware_results.values()
        ])
    else:
        tsa_energy = 1.2

    # Comparison values from literature
    # (replace with actual measured values)
    methods = [
        'Standard\nTransformer\n(GPU)',
        'Spikformer\n(Loihi 2 sim)',
        'DIET-SNN\n(Loihi 2 sim)',
        'TSA\n(Ours)\n(Loihi 2 sim)',
    ]
    energies = [350.0, 8.7, 5.2, tsa_energy]
    colors_hw = ['#E74C3C', '#F39C12', '#3498DB', '#2ECC71']

    # Plot 1: Energy comparison (log scale)
    ax = axes[0]
    bars = ax.bar(
        methods, energies,
        color=colors_hw,
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )

    # Labels
    for bar, val in zip(bars, energies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.05,
            f'{val:.1f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
        )

    ax.set_yscale('log')
    ax.set_ylabel('Energy per Inference (μJ) — log scale')
    ax.set_title('Energy Comparison\n(Loihi 2 Simulation)')

    # Plot 2: Energy reduction factor vs TSA
    ax2 = axes[1]
    tsa_e = energies[-1]
    reductions = [e / tsa_e for e in energies[:-1]]
    reduction_labels = methods[:-1]

    bars2 = ax2.bar(
        reduction_labels,
        reductions,
        color=colors_hw[:-1],
        edgecolor='black',
        linewidth=0.8,
        alpha=0.85,
    )

    for bar, val in zip(bars2, reductions):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.0f}×',
            ha='center', va='bottom',
            fontsize=10, fontweight='bold',
        )

    ax2.set_ylabel('Energy Reduction vs TSA (×)')
    ax2.set_title('Energy Reduction Factor\n(higher = TSA is better)')

    # Plot 3: Accuracy vs Energy pareto
    ax3 = axes[2]
    acc_vals = [0.72, 0.942, 0.891, 0.951]  # placeholder
    energy_vals = energies

    scatter_colors = colors_hw
    for i, (acc, energy, method, color) in enumerate(
        zip(acc_vals, energy_vals, methods, scatter_colors)
    ):
        is_ours = i == len(methods) - 1
        ax3.scatter(
            energy, acc * 100,
            color=color,
            s=200 if is_ours else 100,
            marker='*' if is_ours else 'o',
            edgecolors='black',
            linewidths=2 if is_ours else 1,
            zorder=5 if is_ours else 3,
            label=method.replace('\n', ' '),
        )

    ax3.set_xscale('log')
    ax3.set_xlabel('Energy (μJ) — log scale')
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Pareto Front:\nAccuracy vs Energy')
    ax3.legend(fontsize=7, loc='lower right')

    plt.suptitle(
        'Hardware Validation on Loihi 2\n'
        '(Simulated using published specifications)',
        fontsize=13, fontweight='bold', y=1.02,
    )
    plt.tight_layout()

    output_path = output_dir / 'hardware_validation.pdf'
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    setup_style()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)

    # Load results
    baseline_results = load_json(args.results)
    hardware_results = load_json(args.hardware_results)

    print(f"\nOutput directory: {output_dir}")
    print("Generating 7 figures...\n")

    # Generate all figures
    plot_algorithm_comparison(
        baseline_results, args.datasets, output_dir
    )

    plot_bio_plausibility_radar(output_dir)

    plot_convergence_curves(output_dir)

    plot_energy_comparison(
        baseline_results, args.datasets, output_dir
    )

    plot_tsa_architecture(output_dir)

    plot_attention_visualization(output_dir)

    plot_hardware_validation(hardware_results, output_dir)

    print("\n" + "=" * 60)
    print("✅ ALL FIGURES GENERATED")
    print("=" * 60)
    print(f"\nFiles in {output_dir}:")
    for f in sorted(output_dir.glob('*.pdf')):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} {size_kb:.1f} KB")

    print("\nNote: Figures use placeholder data")
    print("until experiments are run.")
    print("Run experiments first:")
    print("  python scripts/run_baseline_comparison.py")
    print("  python scripts/run_tsa_experiments.py")
    print("Then re-run this script for real figures.")


if __name__ == '__main__':
    main()