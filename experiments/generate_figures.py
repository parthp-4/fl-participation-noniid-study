#!/usr/bin/env python3
"""Generate all 5 publication figures from experiment results."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, '..', 'results', 'all_results.csv')
FIGDIR  = os.path.join(SCRIPT_DIR, '..', 'paper', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

df = pd.read_csv(RESULTS)
print(f"Loaded {len(df)} experiments")

ALPHAS = [0.05, 0.1, 0.5, 1.0]
PARTS  = [0.2, 0.5, 0.8, 1.0]
ALGOS  = ['fedavg', 'fedprox', 'scaffold']
LABELS = {'fedavg': 'FedAvg', 'fedprox': 'FedProx', 'scaffold': 'SCAFFOLD'}
COLORS = {'fedavg': '#2196F3', 'fedprox': '#4CAF50', 'scaffold': '#FF5722'}

plt.rcParams.update({'font.size': 11, 'font.family': 'serif'})

# ── Fig 1: Convergence curves at alpha=0.1 ──
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for i, alg in enumerate(ALGOS):
    ax = axes[i]
    for pr in PARTS:
        row = df[(df['algorithm']==alg) & (df['alpha']==0.1) & (df['participation_rate']==pr)]
        if len(row) == 0: continue
        rr = json.loads(row.iloc[0]['round_results'])
        rounds = [r['round'] for r in rr]
        accs = [r['accuracy']*100 for r in rr]
        ax.plot(rounds, accs, marker='o', markersize=4, linewidth=1.8, label=f'p={int(pr*100)}%')
    ax.set_title(LABELS[alg], fontsize=14, fontweight='bold')
    ax.set_xlabel('Communication Round')
    if i == 0: ax.set_ylabel('Test Accuracy (%)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 100])
plt.suptitle('Convergence Curves at α = 0.1 (Severe Heterogeneity)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig1_convergence_alpha01.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 1: Convergence curves")

# ── Fig 2: Heatmaps ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, alg in enumerate(ALGOS):
    mat = np.zeros((len(PARTS), len(ALPHAS)))
    for r, pr in enumerate(PARTS):
        for c, al in enumerate(ALPHAS):
            row = df[(df['algorithm']==alg) & (df['alpha']==al) & (df['participation_rate']==pr)]
            mat[r, c] = row.iloc[0]['final_accuracy']*100 if len(row) else 0
    sns.heatmap(mat, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[str(a) for a in ALPHAS],
                yticklabels=[f'{int(p*100)}%' for p in PARTS],
                ax=axes[i], vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'})
    axes[i].set_title(LABELS[alg], fontsize=14, fontweight='bold')
    axes[i].set_xlabel('Dirichlet α')
    if i == 0: axes[i].set_ylabel('Participation Rate')
plt.suptitle('Final Test Accuracy Across the Full Parameter Grid', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig2_heatmap_all.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 2: Heatmaps")

# ── Fig 3: Rounds to 80% at p=0.2 ──
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(ALPHAS))
w = 0.25
for i, alg in enumerate(ALGOS):
    vals = []
    for al in ALPHAS:
        row = df[(df['algorithm']==alg) & (df['alpha']==al) & (df['participation_rate']==0.2)]
        v = row.iloc[0]['rounds_to_80pct'] if len(row) else 30
        vals.append(min(v, 30))
    bars = ax.bar(x + i*w, vals, w, label=LABELS[alg], color=COLORS[alg], alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels([str(a) for a in ALPHAS])
ax.set_xlabel('Dirichlet α', fontsize=12)
ax.set_ylabel('Rounds to 80% Accuracy', fontsize=12)
ax.legend(fontsize=10)
ax.set_title('Convergence Speed at 20% Participation Rate', fontsize=13)
ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5)
ax.annotate('Did not converge →', xy=(0.02, 29), fontsize=8, color='gray')
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig3_rounds_to_80.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 3: Rounds to 80%")

# ── Fig 4: Degradation curves ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, al in enumerate([0.05, 0.5]):
    ax = axes[idx]
    for alg in ALGOS:
        accs = []
        for pr in PARTS:
            row = df[(df['algorithm']==alg) & (df['alpha']==al) & (df['participation_rate']==pr)]
            accs.append(row.iloc[0]['final_accuracy']*100 if len(row) else 0)
        ax.plot([int(p*100) for p in PARTS], accs, marker='s', linewidth=2.2,
                color=COLORS[alg], label=LABELS[alg])
    ax.set_xlabel('Participation Rate (%)', fontsize=11)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=11)
    ax.set_title(f'α = {al} ({"Extreme" if al==0.05 else "Moderate"} Heterogeneity)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
plt.suptitle('Algorithm Performance vs. Participation Rate', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig4_participation_degradation.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 4: Degradation curves")

# ── Fig 5: Per-client fairness ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, al in enumerate([0.05, 1.0]):
    ax = axes[idx]
    for alg in ALGOS:
        means, stds = [], []
        for pr in PARTS:
            row = df[(df['algorithm']==alg) & (df['alpha']==al) & (df['participation_rate']==pr)]
            if len(row):
                means.append(row.iloc[0]['per_client_acc_mean']*100)
                stds.append(row.iloc[0]['per_client_acc_std']*100)
            else:
                means.append(0); stds.append(0)
        x = [int(p*100) for p in PARTS]
        ax.errorbar(x, means, yerr=stds, marker='o', linewidth=2, capsize=5,
                    color=COLORS[alg], label=LABELS[alg])
    ax.set_xlabel('Participation Rate (%)', fontsize=11)
    ax.set_ylabel('Per-Client Accuracy (%)', fontsize=11)
    ax.set_title(f'α = {al} ({"Extreme" if al==0.05 else "Mild"} Heterogeneity)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
plt.suptitle('Per-Client Accuracy Fairness (Mean ± Std Dev)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig5_per_client_variance.png'), dpi=200, bbox_inches='tight')
plt.close()
print("✓ Fig 5: Per-client variance")

# ── Print summary table ──
print(f"\n{'='*60}")
print("SUMMARY TABLE — Best algorithm per cell:")
print(f"{'='*60}")
print(f"{'α':>6} {'p':>6} {'Best':>10} {'Acc%':>8}")
print(f"{'-'*6} {'-'*6} {'-'*10} {'-'*8}")
for al in ALPHAS:
    for pr in PARTS:
        sub = df[(df['alpha']==al) & (df['participation_rate']==pr)]
        if len(sub):
            best = sub.loc[sub['final_accuracy'].idxmax()]
            print(f"{al:>6} {pr:>6} {best['algorithm']:>10} {best['final_accuracy']*100:>7.2f}")
print(f"\nAll figures saved to: {FIGDIR}")
