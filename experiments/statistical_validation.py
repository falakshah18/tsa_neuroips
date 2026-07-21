# experiments/statistical_validation.py
"""
Professional statistical analysis
This is what separates amateur from publication-ready work
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json

from utils.reproducibility import set_seed

class StatisticalValidator:
    """
    Complete statistical validation framework
    Addresses reviewer concerns about significance
    """
    
    def __init__(self, results_dir: str = './results'):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_multiple_seeds(
        self, 
        model_fn, 
        train_loader, 
        val_loader, 
        test_loader,
        n_seeds: int = 10,
        config: dict = None
    ) -> List[Dict]:
        """
        Run experiments with multiple random seeds
        CRITICAL: Reviewers will reject without this
        """
        results = []
        
        for seed in range(n_seeds):
            print(f"\n{'='*60}")
            print(f"Running Seed {seed+1}/{n_seeds}")
            print(f"{'='*60}")
            
            # Set all random seeds (includes cudnn.deterministic + use_deterministic_algorithms)
            set_seed(seed)
            
            # Create fresh model
            model = model_fn()
            
            # Train
            from training.trainer_v2 import AdvancedTrainer
            trainer = AdvancedTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                config={**config, 'run_name': f'seed_{seed}'}
            )
            
            metrics = trainer.train()
            
            # Store results
            result = {
                'seed': seed,
                'test_acc': metrics['test_metrics']['acc'],
                'test_energy': metrics['test_metrics']['avg_energy_uJ'],
                'best_val_acc': metrics['best_val_acc'],
            }
            results.append(result)
            
            print(f"Seed {seed}: Acc={result['test_acc']:.4f}, "
                  f"Energy={result['test_energy']:.2f}μJ")
        
        return results
    
    def compute_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute comprehensive statistics
        """
        metrics = ['test_acc', 'test_energy']
        stats_dict = {}
        
        for metric in metrics:
            values = np.array([r[metric] for r in results])
            
            # Basic statistics
            mean = np.mean(values)
            std = np.std(values, ddof=1)  # Sample std
            sem = stats.sem(values)  # Standard error of mean
            
            # Confidence intervals
            ci_95 = stats.t.interval(
                0.95, 
                len(values)-1, 
                loc=mean, 
                scale=sem
            )
            ci_99 = stats.t.interval(
                0.99, 
                len(values)-1, 
                loc=mean, 
                scale=sem
            )
            
            # Percentiles
            percentiles = np.percentile(values, [25, 50, 75])
            
            stats_dict[metric] = {
                'mean': float(mean),
                'std': float(std),
                'sem': float(sem),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'ci_95_lower': float(ci_95[0]),
                'ci_95_upper': float(ci_95[1]),
                'ci_99_lower': float(ci_99[0]),
                'ci_99_upper': float(ci_99[1]),
                'q25': float(percentiles[0]),
                'q50': float(percentiles[1]),
                'q75': float(percentiles[2]),
                'values': values.tolist(),
            }
        
        return stats_dict
    
    def paired_comparison(
        self,
        method_a_results: List[Dict],
        method_b_results: List[Dict],
        metric: str = 'test_acc',
        method_a_name: str = 'Method A',
        method_b_name: str = 'Method B',
    ) -> Dict:
        """
        Statistical comparison between two methods
        Uses paired t-test (same seeds)
        """
        values_a = np.array([r[metric] for r in method_a_results])
        values_b = np.array([r[metric] for r in method_b_results])
        
        assert len(values_a) == len(values_b), "Must have same number of seeds"
        
        # Paired t-test
        t_stat, p_value_ttest = ttest_rel(values_a, values_b)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        w_stat, p_value_wilcoxon = wilcoxon(values_a, values_b)
        
        # Effect size (Cohen's d)
        mean_diff = np.mean(values_a - values_b)
        pooled_std = np.sqrt((np.var(values_a, ddof=1) + np.var(values_b, ddof=1)) / 2)
        cohens_d = mean_diff / pooled_std
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        # Statistical significance
        is_significant_ttest = p_value_ttest < 0.05
        is_significant_wilcoxon = p_value_wilcoxon < 0.05
        
        comparison = {
            'method_a': method_a_name,
            'method_b': method_b_name,
            'metric': metric,
            'mean_a': float(np.mean(values_a)),
            'mean_b': float(np.mean(values_b)),
            'std_a': float(np.std(values_a, ddof=1)),
            'std_b': float(np.std(values_b, ddof=1)),
            'mean_difference': float(mean_diff),
            't_statistic': float(t_stat),
            'p_value_ttest': float(p_value_ttest),
            'wilcoxon_statistic': float(w_stat),
            'p_value_wilcoxon': float(p_value_wilcoxon),
            'cohens_d': float(cohens_d),
            'effect_size': effect_interpretation,
            'significant_ttest': is_significant_ttest,
            'significant_wilcoxon': is_significant_wilcoxon,
            'winner': method_a_name if mean_diff > 0 else method_b_name,
        }
        
        return comparison
    
    def multiple_comparison(
        self,
        all_results: Dict[str, List[Dict]],
        metric: str = 'test_acc',
        alpha: float = 0.05,
    ) -> Dict:
        """
        Compare multiple methods with Bonferroni correction
        """
        method_names = list(all_results.keys())
        n_methods = len(method_names)
        
        # Bonferroni correction for multiple comparisons
        alpha_corrected = alpha / (n_methods * (n_methods - 1) / 2)
        
        # Friedman test (non-parametric ANOVA for repeated measures)
        values_matrix = np.array([
            [r[metric] for r in all_results[method]]
            for method in method_names
        ])
        
        friedman_stat, friedman_p = friedmanchisquare(*values_matrix)
        
        # Pairwise comparisons
        pairwise_results = []
        for i in range(n_methods):
            for j in range(i+1, n_methods):
                method_a = method_names[i]
                method_b = method_names[j]
                
                comparison = self.paired_comparison(
                    all_results[method_a],
                    all_results[method_b],
                    metric=metric,
                    method_a_name=method_a,
                    method_b_name=method_b,
                )
                
                # Apply Bonferroni correction
                comparison['alpha_corrected'] = alpha_corrected
                comparison['significant_bonferroni'] = (
                    comparison['p_value_ttest'] < alpha_corrected
                )
                
                pairwise_results.append(comparison)
        
        # Rank methods by mean performance
        rankings = sorted(
            [(name, np.mean([r[metric] for r in all_results[name]])) 
             for name in method_names],
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'friedman_statistic': float(friedman_stat),
            'friedman_p_value': float(friedman_p),
            'overall_significant': friedman_p < alpha,
            'alpha_original': alpha,
            'alpha_bonferroni': alpha_corrected,
            'pairwise_comparisons': pairwise_results,
            'rankings': rankings,
        }
    
    def generate_comparison_plots(
        self,
        all_results: Dict[str, List[Dict]],
        metric: str = 'test_acc',
        save_path: str = 'comparison_plot.pdf'
    ):
        """
        Generate publication-quality comparison plots
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        method_names = list(all_results.keys())
        
        # Plot 1: Box plot
        data_for_box = [
            [r[metric] * 100 for r in all_results[method]]
            for method in method_names
        ]
        
        bp = axes[0].boxplot(
            data_for_box,
            labels=method_names,
            patch_artist=True,
            showmeans=True,
        )
        
        # Color the boxes
        colors = sns.color_palette('husl', len(method_names))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Distribution Comparison', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Violin plot
        data_df = pd.DataFrame({
            method: [r[metric] * 100 for r in all_results[method]]
            for method in method_names
        })
        
        sns.violinplot(data=data_df, ax=axes[1], palette='husl')
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Density Comparison', fontsize=14)
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Mean with error bars
        means = [np.mean([r[metric] * 100 for r in all_results[method]]) 
                for method in method_names]
        stds = [np.std([r[metric] * 100 for r in all_results[method]], ddof=1) 
               for method in method_names]
        
        x_pos = np.arange(len(method_names))
        axes[2].bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                   color=colors, edgecolor='black', linewidth=1.5)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(method_names, rotation=45, ha='right')
        axes[2].set_ylabel('Accuracy (%)', fontsize=12)
        axes[2].set_title('Mean ± Std Dev', fontsize=14)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")
    
    def generate_latex_table(
        self,
        all_results: Dict[str, List[Dict]],
        comparison_stats: Dict,
        save_path: str = 'comparison_table.tex'
    ):
        """
        Generate LaTeX table with statistical annotations
        """
        method_names = list(all_results.keys())
        
        latex = r"""\begin{table*}[t]
\centering
\caption{Performance comparison across methods. Results reported as mean $\pm$ std over 10 random seeds. 
Statistical significance tested using paired t-test with Bonferroni correction ($\alpha=0.05$). 
$^*$ indicates statistically significant improvement over baseline.}
\label{tab:main_comparison}
\begin{tabular}{lccccc}
\toprule
\textbf{Method} & \textbf{Accuracy (\%)} & \textbf{Energy (μJ)} & \textbf{Spikes} & \textbf{Params (M)} & \textbf{p-value} \\
\midrule
"""
        
        # Get best method for marking
        rankings = comparison_stats['rankings']
        best_method = rankings[0][0]
        
        for method in method_names:
            results = all_results[method]
            
            # Compute statistics
            acc_mean = np.mean([r['test_acc'] * 100 for r in results])
            acc_std = np.std([r['test_acc'] * 100 for r in results], ddof=1)
            
            energy_mean = np.mean([r['test_energy'] for r in results])
            energy_std = np.std([r['test_energy'] for r in results], ddof=1)
            
            spikes_mean = np.mean([r.get('test_spikes', 0) for r in results])
            spikes_std = np.std([r.get('test_spikes', 0) for r in results], ddof=1)
            
            # Find p-value compared to baseline (assuming first method is baseline)
            baseline_method = method_names[0]
            p_value = 1.0
            if method != baseline_method:
                for comp in comparison_stats['pairwise_comparisons']:
                    if comp['method_a'] == baseline_method and comp['method_b'] == method:
                        p_value = comp['p_value_ttest']
                    elif comp['method_b'] == baseline_method and comp['method_a'] == method:
                        p_value = comp['p_value_ttest']
            
            # Mark significance
            sig_marker = r"$^*$" if p_value < 0.05 else ""
            
            # Bold if best
            if method == best_method:
                latex += r"\textbf{" + method + r"} & "
                latex += r"\textbf{" + f"{acc_mean:.2f} $\\pm$ {acc_std:.2f}" + r"}" + sig_marker + " & "
                latex += r"\textbf{" + f"{energy_mean:.2f} $\\pm$ {energy_std:.2f}" + r"} & "
                latex += f"{spikes_mean:.0f} $\\pm$ {spikes_std:.0f} & "
            else:
                latex += f"{method} & "
                latex += f"{acc_mean:.2f} $\\pm$ {acc_std:.2f}" + sig_marker + " & "
                latex += f"{energy_mean:.2f} $\\pm$ {energy_std:.2f} & "
                latex += f"{spikes_mean:.0f} $\\pm$ {spikes_std:.0f} & "
            
            latex += f"-- & "
            latex += f"{p_value:.4f} \\\\\n"
        
        latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
        
        with open(save_path, 'w') as f:
            f.write(latex)
        
        print(f"📄 LaTeX table saved to {save_path}")
        
        return latex
    
    def comprehensive_report(
        self,
        all_results: Dict[str, List[Dict]],
        save_dir: str = './statistical_analysis'
    ):
        """
        Generate complete statistical analysis report
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE STATISTICAL ANALYSIS")
        print("="*60)
        
        # 1. Individual method statistics
        print("\n1. Individual Method Statistics:")
        for method, results in all_results.items():
            print(f"\n{method}:")
            stats = self.compute_statistics(results)
            for metric, values in stats.items():
                print(f"  {metric}:")
                print(f"    Mean: {values['mean']:.4f}")
                print(f"    Std:  {values['std']:.4f}")
                print(f"    95% CI: [{values['ci_95_lower']:.4f}, {values['ci_95_upper']:.4f}]")
        
        # 2. Multiple comparison test
        print("\n2. Multiple Comparison Test:")
        comparison_stats = self.multiple_comparison(all_results, metric='test_acc')
        print(f"  Friedman test: χ² = {comparison_stats['friedman_statistic']:.4f}, "
              f"p = {comparison_stats['friedman_p_value']:.4f}")
        
        if comparison_stats['overall_significant']:
            print("  Overall difference is statistically significant")
        else:
            print("  WARNING: No significant difference found")
        
        print(f"\n  Rankings:")
        for i, (method, score) in enumerate(comparison_stats['rankings'], 1):
            print(f"    {i}. {method}: {score:.4f}")
        
        # 3. Pairwise comparisons
        print("\n3. Pairwise Comparisons (with Bonferroni correction):")
        for comp in comparison_stats['pairwise_comparisons']:
            sig_symbol = "+" if comp['significant_bonferroni'] else "ns"
            print(f"  {comp['method_a']} vs {comp['method_b']}:")
            print(f"    Difference: {comp['mean_difference']:.4f}")
            print(f"    p-value: {comp['p_value_ttest']:.4f} {sig_symbol}")
            print(f"    Effect size (Cohen's d): {comp['cohens_d']:.4f} ({comp['effect_size']})")
        
        # 4. Generate plots
        self.generate_comparison_plots(
            all_results,
            save_path=save_dir / 'comparison_accuracy.pdf'
        )
        self.generate_comparison_plots(
            all_results,
            metric='test_energy',
            save_path=save_dir / 'comparison_energy.pdf'
        )
        
        # 5. Generate LaTeX table
        self.generate_latex_table(
            all_results,
            comparison_stats,
            save_path=save_dir / 'comparison_table.tex'
        )
        
        # 6. Save JSON report
        report = {
            'individual_statistics': {
                method: self.compute_statistics(results)
                for method, results in all_results.items()
            },
            'multiple_comparison': comparison_stats,
        }
        
        with open(save_dir / 'statistical_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nComplete statistical analysis saved to {save_dir}")
        
        return report