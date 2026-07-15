#!/usr/bin/env python3
"""
Generate Virtual Screening figures from results.
Creates comprehensive figures for the DUD-E benchmark evaluation.
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 13,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "results")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    scores_path = os.path.join(OUT_DIR, "vs_compound_scores.json")
    if not os.path.exists(scores_path):
        print(f"ERROR: {scores_path} not found.")
        sys.exit(1)
    with open(scores_path) as f:
        scores_data = json.load(f)
    y_true = np.array([1 if s["is_active"] else 0 for s in scores_data])
    y_score = np.array([s["predicted_pKd"] for s in scores_data])
    valid = ~np.isnan(y_score) & ~np.isinf(y_score)
    return y_true[valid], y_score[valid]


def fig1_roc_pr(y_true, y_score):
    print("Fig 1: ROC & PR Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'MMDCG-DTA (AUC={roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.fill_between(fpr, tpr, alpha=0.15, color='blue')
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
    ax1.set_title(f'ROC Curve — AUC={roc_auc:.4f}', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2)

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    baseline = np.sum(y_true) / len(y_true)
    ax2.plot(recall, precision, 'r-', linewidth=2.5, label=f'MMDCG-DTA (PR-AUC={pr_auc:.4f})')
    ax2.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
    ax2.set_title(f'PR Curve — AUC={pr_auc:.4f}', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.2)

    fig.savefig(os.path.join(FIG_DIR, 'fig1_roc_pr.png'), facecolor='white')
    fig.savefig(os.path.join(FIG_DIR, 'fig1_roc_pr.pdf'), facecolor='white')
    plt.close()


def fig2_distribution(y_true, y_score):
    print("Fig 2: Score Distribution...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    active_scores = y_score[y_true == 1]
    decoy_scores = y_score[y_true == 0]

    ax1.hist(decoy_scores, bins=50, alpha=0.6, color='#2196F3', density=True,
             label=f'Decoys (n={len(decoy_scores)})')
    ax1.hist(active_scores, bins=50, alpha=0.7, color='#FF5722', density=True,
             label=f'Actives (n={len(active_scores)})')
    ax1.set_xlabel('Predicted Score'); ax1.set_ylabel('Density')
    ax1.set_title('Score Distribution', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.2)

    vp = ax2.violinplot([decoy_scores, active_scores], positions=[1, 2],
                          showmeans=True, showmedians=True)
    for body, color in zip(vp['bodies'], ['#2196F3', '#FF5722']):
        body.set_facecolor(color); body.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels([f'Decoys\n(n={len(decoy_scores)})', f'Actives\n(n={len(active_scores)})'])
    ax2.set_ylabel('Predicted Score')
    ax2.set_title('Score Comparison', fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='y')

    fig.savefig(os.path.join(FIG_DIR, 'fig2_distribution.png'), facecolor='white')
    fig.savefig(os.path.join(FIG_DIR, 'fig2_distribution.pdf'), facecolor='white')
    plt.close()


def fig3_enrichment(y_true, y_score):
    print("Fig 3: Enrichment Analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    sorted_idx = np.argsort(y_score)[::-1]
    sorted_labels = y_true[sorted_idx]
    n_total = len(y_true); n_actives = int(np.sum(y_true))

    ef_pcts = [0.5, 1, 2, 5, 10]
    ef_values = []
    for pct in ef_pcts:
        top_n = max(1, int(n_total * pct / 100))
        ef = (np.sum(sorted_labels[:top_n]) / top_n) / (n_actives / n_total)
        ef_values.append(ef)

    colors = ['#1B5E20', '#2E7D32', '#388E3C', '#4CAF50', '#81C784']
    bars = ax1.bar(range(len(ef_pcts)), ef_values, color=colors, edgecolor='white')
    ax1.set_xticks(range(len(ef_pcts)))
    ax1.set_xticklabels([f'Top {p}%' for p in ef_pcts])
    ax1.set_ylabel('Enrichment Factor'); ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Enrichment Factor', fontweight='bold')
    for bar, ef in zip(bars, ef_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{ef:.1f}x', ha='center', fontweight='bold')
    ax1.grid(True, alpha=0.2, axis='y')

    cumulative = np.cumsum(sorted_labels)
    x_pct = np.arange(1, n_total + 1) / n_total * 100
    ax2.plot(x_pct, cumulative / n_actives * 100, 'b-', linewidth=2.5, label='MMDCG-DTA')
    ax2.plot([0, 100], [0, 100], 'k--', alpha=0.3, label='Random')
    ax2.set_xlabel('Fraction Screened (%)'); ax2.set_ylabel('Actives Found (%)')
    ax2.set_title('Cumulative Enrichment', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.2)

    fig.savefig(os.path.join(FIG_DIR, 'fig3_enrichment.png'), facecolor='white')
    fig.savefig(os.path.join(FIG_DIR, 'fig3_enrichment.pdf'), facecolor='white')
    plt.close()


if __name__ == "__main__":
    print("MMDCG-DTA VS Figure Generation")
    y_true, y_score = load_data()
    print(f"Loaded {len(y_true)} compounds ({int(np.sum(y_true))} actives)")
    fig1_roc_pr(y_true, y_score)
    fig2_distribution(y_true, y_score)
    fig3_enrichment(y_true, y_score)
    print(f"All figures saved to {FIG_DIR}/")
