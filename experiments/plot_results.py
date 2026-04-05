"""
Generate all publication-quality figures from experiment results.

Usage:
    python plot_results.py --results path/to/all_results.csv --outdir paper/figures/
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ALGORITHMS = ['fedavg', 'fedprox', 'scaffold']
ALPHAS = [0.05, 0.1, 0.5, 1.0]
PART_RATES = [0.2, 0.5, 0.8, 1.0]

# Publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def load_results(path):
    df = pd.read_csv(path)
    df['round_results'] = df['round_results'].apply(json.loads)
    return df


def fig1_convergence(df, outdir):
    """Convergence curves at alpha=0.1, 100% participation."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ALGORITHMS:
        row = df[(df['algorithm'] == alg) &
                 (df['alpha'] == 0.1) &
                 (df['participation_rate'] == 1.0)].iloc[0]
        rr = row['round_results']
        ax.plot([r['round'] for r in rr],
                [r['accuracy'] for r in rr],
                label=alg.upper(), linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Convergence Under Non-IID (α=0.1, Full Participation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig1_convergence_alpha01.png', bbox_inches='tight')
    plt.close()


def fig2_heatmaps(df, outdir):
    """Side-by-side accuracy heatmaps for all three algorithms."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, alg in enumerate(ALGORITHMS):
        pivot = df[df['algorithm'] == alg].pivot(
            index='participation_rate',
            columns='alpha',
            values='final_accuracy'
        )
        sns.heatmap(pivot, ax=axes[i], annot=True, fmt='.3f',
                    cmap='RdYlGn', vmin=0.5, vmax=1.0,
                    cbar_kws={'label': 'Final Accuracy'})
        axes[i].set_title(alg.upper())
        axes[i].set_xlabel('Dirichlet α')
        axes[i].set_ylabel('Participation Rate')
    plt.suptitle(
        'Final Accuracy: Algorithm × Participation Rate × Non-IID Severity',
        fontsize=14, y=1.02
    )
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig2_heatmap_all.png', bbox_inches='tight')
    plt.close()


def fig3_rounds_to_target(df, outdir):
    """Grouped bar chart: rounds to reach 80% at 20% participation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.25
    x = np.arange(len(ALPHAS))
    for j, alg in enumerate(ALGORITHMS):
        vals = []
        for alpha in ALPHAS:
            row = df[(df['algorithm'] == alg) &
                     (df['alpha'] == alpha) &
                     (df['participation_rate'] == 0.2)]
            v = row['rounds_to_80pct'].values[0] if len(row) > 0 else 999
            vals.append(min(v, 50))
        ax.bar(x + j * width, vals, width, label=alg.upper(), alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'α={a}' for a in ALPHAS])
    ax.set_ylabel('Rounds to Reach 80% Accuracy')
    ax.set_title('Convergence Speed Under 20% Participation')
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Max rounds')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig3_rounds_to_80.png', bbox_inches='tight')
    plt.close()


def fig4_participation_degradation(df, outdir):
    """Accuracy vs participation rate at extreme alphas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, alpha in zip(axes, [0.05, 0.5]):
        for alg in ALGORITHMS:
            accs = []
            for part in PART_RATES:
                row = df[(df['algorithm'] == alg) &
                         (df['alpha'] == alpha) &
                         (df['participation_rate'] == part)]
                accs.append(row['final_accuracy'].values[0] if len(row) > 0 else 0)
            ax.plot(PART_RATES, accs, marker='o', label=alg.upper(), linewidth=2)
        ax.set_xlabel('Participation Rate')
        ax.set_ylabel('Final Accuracy')
        ax.set_title(f'Accuracy vs Participation (α={alpha})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig4_participation_degradation.png', bbox_inches='tight')
    plt.close()


def fig5_client_variance(df, outdir):
    """Per-client accuracy variance at 50% participation."""
    fig, ax = plt.subplots(figsize=(10, 5))
    labels, means, stds = [], [], []
    for alg in ALGORITHMS:
        for alpha in [0.05, 1.0]:
            row = df[(df['algorithm'] == alg) &
                     (df['alpha'] == alpha) &
                     (df['participation_rate'] == 0.5)]
            if len(row) > 0:
                labels.append(f'{alg.upper()}\nα={alpha}')
                means.append(row['per_client_acc_mean'].values[0])
                stds.append(row['per_client_acc_std'].values[0])
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=5, alpha=0.85,
           color=sns.color_palette('Set2', len(labels)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Per-Client Accuracy (mean ± std)')
    ax.set_title('Per-Client Accuracy Variance (50% Participation)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f'{outdir}/fig5_per_client_variance.png', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to all_results.csv')
    parser.add_argument('--outdir', default='../paper/figures/', help='Output directory')
    args = parser.parse_args()

    df = load_results(args.results)
    fig1_convergence(df, args.outdir)
    fig2_heatmaps(df, args.outdir)
    fig3_rounds_to_target(df, args.outdir)
    fig4_participation_degradation(df, args.outdir)
    fig5_client_variance(df, args.outdir)
    print(f'All figures saved to {args.outdir}')
