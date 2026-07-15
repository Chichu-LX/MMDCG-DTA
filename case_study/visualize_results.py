#!/usr/bin/env python3
"""
Case Study Visualization — Generate Comprehensive Figures
===========================================================
Creates figures demonstrating the three core claims:
  Fig1: Prediction Performance (scatter, error dist, per-complex)
  Fig2: Molecular Mechanics Evidence (energy heatmap, correlations)
  Fig3: Graph Transformation & Edge Reconstruction (HIL changes, edge stats)
  Fig4: Affinity-Stratified Patterns (radar, stratified errors)
"""

import pickle, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

C_LIG = '#E91E63'; C_PROT = '#3F51B5'; C_GNN = '#2196F3'
C_PHYS = '#FF5722'; C_FUSION = '#4CAF50'; C_INTER = '#FF9800'


def load_data():
    reps_path = os.path.join(SCRIPT_DIR, "case_study_representations.pkl")
    results_path = os.path.join(SCRIPT_DIR, "case_study_results.json")

    with open(reps_path, 'rb') as f:
        reps = pickle.load(f)

    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    return reps, results


def fig1_prediction_performance(reps):
    """Prediction overview."""
    print("Fig 1: Prediction Performance...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    cids = sorted(reps.keys())
    yt = np.array([reps[c]['true_pKd'] for c in cids])
    yp = np.array([reps[c].get('predicted_pKd_s1', 0) for c in cids])
    errors = yp - yt

    # Panel A: Scatter
    ax = axes[0]
    r, _ = stats.pearsonr(yt, yp)
    rmse = np.sqrt(np.mean(errors**2))
    sc = ax.scatter(yt, yp, c=np.abs(errors), cmap='RdYlGn_r', s=60,
                    edgecolors='white', linewidth=0.5, alpha=0.85)
    ax.plot([2, 13], [2, 13], 'k--', alpha=0.3)
    ax.set_xlabel('Experimental pKd', fontweight='bold')
    ax.set_ylabel('Predicted pKd', fontweight='bold')
    ax.set_title(f'True vs Predicted\nPearson r={r:.4f}, RMSE={rmse:.4f}', fontweight='bold')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label='|Error|')

    # Panel B: Error distribution
    ax = axes[1]
    ax.hist(errors, bins=20, edgecolor='white', color=C_GNN, alpha=0.8, density=True)
    x = np.linspace(errors.min(), errors.max(), 200)
    mu, sigma = np.mean(errors), np.std(errors)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'N({mu:.3f}, {sigma:.3f})')
    ax.axvline(0, color='black', linestyle='--', alpha=0.4)
    ax.set_xlabel('Error (pKd)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Error Distribution', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel C: Per-complex
    ax = axes[2]
    sort_idx = np.argsort(yt)
    ax.plot(range(len(cids)), yt[sort_idx], 'o-', color=C_PROT, markersize=4, label='True')
    ax.plot(range(len(cids)), yp[sort_idx], 's-', color=C_LIG, markersize=4, label='Predicted')
    ax.fill_between(range(len(cids)), yt[sort_idx], yp[sort_idx], alpha=0.15, color='red')
    ax.set_xlabel('Complex (sorted)', fontweight='bold')
    ax.set_ylabel('pKd', fontweight='bold')
    ax.set_title('Per-Complex Predictions', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    fig.savefig(os.path.join(FIG_DIR, 'fig1_prediction_performance.png'), facecolor='white')
    plt.close()


def fig2_molecular_mechanics(reps):
    """Energy analysis demonstrating claim A."""
    print("Fig 2: Molecular Mechanics Evidence...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    cids = sorted(reps.keys())
    yt = np.array([reps[c]['true_pKd'] for c in cids])

    energy_keys = ['L_bond', 'P_bond', 'L_ang', 'L_tor', 'P_ang', 'P_tor',
                   'I_vdw', 'I_elec', 'I_hb']
    energy_labels = ['Lig Bond', 'Prot Bond', 'Lig Angle', 'Lig Torsion',
                     'Prot Angle', 'Prot Torsion', 'VDW', 'Electrostatic', 'H-Bond']

    # Panel A: Energy heatmap
    ax = axes[0, 0]
    sort_by_pkd = np.argsort(yt)
    energy_matrix = np.column_stack([
        (np.array([reps[c]['energies'].get(k, 0) for c in cids]) -
         np.mean([reps[c]['energies'].get(k, 0) for c in cids])) /
        (np.std([reps[c]['energies'].get(k, 0) for c in cids]) + 1e-8)
        for k in energy_keys
    ])[sort_by_pkd]
    im = ax.imshow(energy_matrix.T, aspect='auto', cmap='RdBu_r', interpolation='bilinear')
    ax.set_yticks(range(len(energy_labels)))
    ax.set_yticklabels(energy_labels, fontsize=9)
    ax.set_xlabel('Complex (sorted by pKd)', fontweight='bold')
    ax.set_title('Energy Landscape (Z-score normalized)', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Z-score')

    # Panel B: Energy-pKd correlations
    ax = axes[0, 1]
    corrs = []
    for k in energy_keys:
        vals = np.array([reps[c]['energies'].get(k, 0) for c in cids])
        if np.std(vals) > 1e-8:
            r_val, _ = stats.pearsonr(vals, yt)
        else:
            r_val = 0
        corrs.append(r_val)
    colors = ['#FF5722' if c > 0 else '#2196F3' for c in corrs]
    bars = ax.barh(range(len(energy_labels)), corrs, color=colors, edgecolor='white')
    ax.set_yticks(range(len(energy_labels)))
    ax.set_yticklabels(energy_labels, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Pearson r with pKd', fontweight='bold')
    ax.set_title('Energy-pKd Correlations', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    for bar, c in zip(bars, corrs):
        ax.text(c + 0.02*np.sign(c), bar.get_y()+bar.get_height()/2,
                f'{c:.3f}', va='center', fontsize=8, fontweight='bold')

    # Panel C: Inter-molecular energies vs pKd
    ax = axes[1, 0]
    inter_keys = ['I_vdw', 'I_elec', 'I_hb']
    inter_colors = ['#E91E63', '#3F51B5', '#9C27B0']
    for key, color in zip(inter_keys, inter_colors):
        vals = np.array([reps[c]['energies'].get(key, 0) for c in cids])
        ax.scatter(yt, vals, alpha=0.6, s=30, label=key, color=color, edgecolors='white')
        if np.std(vals) > 1e-8:
            z = np.polyfit(yt, vals, 1)
            ax.plot(np.sort(yt), np.poly1d(z)(np.sort(yt)), '-', color=color, linewidth=1.5)
    ax.set_xlabel('Experimental pKd', fontweight='bold')
    ax.set_ylabel('Energy', fontweight='bold')
    ax.set_title('Inter-Molecular Energies vs Affinity', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

    # Panel D: Total interaction vs pKd
    ax = axes[1, 1]
    total = np.array([abs(reps[c]['energies'].get('I_vdw', 0)) +
                       abs(reps[c]['energies'].get('I_elec', 0)) +
                       abs(reps[c]['energies'].get('I_hb', 0))
                       for c in cids])
    r_val, p_val = stats.pearsonr(total, yt)
    ax.scatter(yt, total, c=C_INTER, s=60, edgecolors='white', alpha=0.8)
    z = np.polyfit(yt, total, 1)
    ax.plot(np.sort(yt), np.poly1d(z)(np.sort(yt)), 'r--', linewidth=1.5,
            label=f'r={r_val:.3f}, p={p_val:.2e}')
    ax.set_xlabel('Experimental pKd', fontweight='bold')
    ax.set_ylabel('Total Interaction Energy', fontweight='bold')
    ax.set_title('Total Interaction vs Affinity', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    sig_count = sum(1 for k in energy_keys
                    if abs(stats.pearsonr([reps[c]['energies'].get(k, 0) for c in cids], yt)[1]) < 0.05
                    if np.std([reps[c]['energies'].get(k, 0) for c in cids]) > 1e-8)
    fig.suptitle(f'Claim A: Molecular Mechanics Learned Physical Essence\n'
                 f'{sig_count}/{len(energy_keys)} energy terms significantly correlated',
                 fontweight='bold', fontsize=14)

    fig.savefig(os.path.join(FIG_DIR, 'fig2_molecular_mechanics.png'), facecolor='white')
    plt.close()


def fig3_hil_edge_analysis(reps):
    """HIL changes and edge reconstruction."""
    print("Fig 3: HIL & Edge Reconstruction...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    cids = sorted(reps.keys())
    yt = np.array([reps[c]['true_pKd'] for c in cids])

    # Panel A: HIL change distribution
    ax = axes[0]
    hil_keys = ['atom_lig_hil_change', 'atom_prot_hil_change',
                'sub_lig_hil_change', 'sub_prot_hil_change']
    hil_labels = ['Atom Lig\nHIL', 'Atom Prot\nHIL', 'Sub Lig\nHIL', 'Sub Prot\nHIL']
    hil_colors = [C_LIG, C_PROT, C_FUSION, C_INTER]

    all_hil = []
    for key in hil_keys:
        vals = [reps[c].get('hil_changes_s1', {}).get(key, 0) for c in cids]
        all_hil.append(vals)

    bp = ax.boxplot(all_hil, patch_artist=True, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=5))
    for patch, color in zip(bp['boxes'], hil_colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_xticklabels(hil_labels, fontsize=9)
    ax.set_ylabel('L2 Norm Change', fontweight='bold')
    ax.set_title('HIL Information Flow Magnitude\n(Representation Change)', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Panel B: HIL change correlation with pKd
    ax = axes[1]
    for key, label, color in zip(hil_keys, hil_labels, hil_colors):
        vals = np.array([reps[c].get('hil_changes_s1', {}).get(key, 0) for c in cids])
        if np.std(vals) > 1e-8:
            r_val, _ = stats.pearsonr(vals, yt)
            ax.barh(hil_labels.index(label), r_val, color=color, alpha=0.7,
                    label=f'{label} (r={r_val:.3f})')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_yticks(range(len(hil_labels)))
    ax.set_yticklabels(hil_labels, fontsize=9)
    ax.set_xlabel('Pearson r with pKd', fontweight='bold')
    ax.set_title('HIL Change ~ Affinity Correlation', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.2, axis='x')

    # Panel C: Affinity-stratified HIL changes
    ax = axes[2]
    low = yt < 7; high = yt >= 9; med = ~low & ~high
    x_pos = np.arange(len(hil_keys))
    width = 0.25
    for mask, pos_shift, label, hatch in [
        (low, -width, 'Low (pKd<7)', '//'),
        (med, 0, 'Med (7-9)', ''),
        (high, width, 'High (pKd>9)', '\\\\'),
    ]:
        means = []
        for key in hil_keys:
            vals = [reps[c].get('hil_changes_s1', {}).get(key, 0) for c in cids]
            means.append(np.mean([v for i, v in enumerate(vals) if mask[i]]))
        ax.bar(x_pos + pos_shift, means, width, label=label, alpha=0.7, hatch=hatch)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hil_labels, fontsize=9)
    ax.set_ylabel('Mean HIL Change', fontweight='bold')
    ax.set_title('HIL Changes by Affinity Range', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2, axis='y')

    fig.suptitle('Claim B: Graph Transformation & Information Flow',
                 fontweight='bold', fontsize=14)
    fig.savefig(os.path.join(FIG_DIR, 'fig3_hil_edge_analysis.png'), facecolor='white')
    plt.close()


def fig4_affinity_patterns(reps):
    """Satisfactory patterns across affinity ranges."""
    print("Fig 4: Affinity Patterns...")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    cids = sorted(reps.keys())
    yt = np.array([reps[c]['true_pKd'] for c in cids])
    low = yt < 7; high = yt >= 9; med = ~low & ~high

    # Panel A: Stratified energy radar
    ax = axes[0]
    radar_keys = ['I_vdw', 'I_elec', 'I_hb', 'L_bond', 'L_tor']
    radar_labels = ['VDW', 'Elec', 'H-Bond', 'L-Bond', 'L-Tors']

    angles = np.linspace(0, 2*np.pi, len(radar_keys), endpoint=False).tolist()
    angles += angles[:1]

    for mask, color, label in [(low, '#2196F3', 'Low'), (med, '#FF9800', 'Med'), (high, '#F44336', 'High')]:
        means = []
        for k in radar_keys:
            vals = [reps[c]['energies'].get(k, 0) for c in cids]
            means.append(np.mean([v for i, v in enumerate(vals) if mask[i]]))
        means = np.array(means)
        means = (means - means.min()) / (means.max() - means.min() + 1e-8)
        means = np.append(means, means[0])
        ax.fill(angles, means, alpha=0.15, color=color)
        ax.plot(angles, means, 'o-', linewidth=2, color=color, label=label)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_title('Energy Profiles by Affinity', fontweight='bold')
    ax.legend(fontsize=9)

    # Panel B: Error by affinity
    ax = axes[1]
    yp = np.array([reps[c].get('predicted_pKd_s1', 0) for c in cids])
    errors = yp - yt
    error_data = [errors[low], errors[med], errors[high]]
    bp = ax.boxplot(error_data, patch_artist=True, showmeans=True)
    colors = ['#2196F3', '#FF9800', '#F44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.axhline(0, color='black', linestyle='--', alpha=0.4)
    ax.set_xticklabels([f'Low\n(n={low.sum()})', f'Med\n(n={med.sum()})', f'High\n(n={high.sum()})'])
    ax.set_ylabel('Prediction Error (pKd)', fontweight='bold')
    ax.set_title('Error by Affinity Range', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Panel C: Feature correlation ranking
    ax = axes[2]
    all_features = {}
    for key in ['I_vdw', 'I_elec', 'I_hb', 'L_bond', 'L_tor', 'P_bond', 'P_tor']:
        vals = np.array([reps[c]['energies'].get(key, 0) for c in cids])
        if np.std(vals) > 1e-8:
            r_val, p_val = stats.pearsonr(vals, yt)
            all_features[key] = (abs(r_val), r_val, p_val)

    for key in ['atom_prot_hil_change', 'atom_lig_hil_change',
                'sub_lig_hil_change', 'sub_prot_hil_change']:
        vals = np.array([reps[c].get('hil_changes_s1', {}).get(key, 0) for c in cids])
        if np.std(vals) > 1e-8:
            r_val, p_val = stats.pearsonr(vals, yt)
            all_features[key] = (abs(r_val), r_val, p_val)

    sorted_features = sorted(all_features.items(), key=lambda x: x[1][0], reverse=True)[:10]
    names = [f[0].replace('_', ' ').title() for f in sorted_features]
    corrs = [f[1][1] for f in sorted_features]
    feat_colors = ['#FF5722' if c > 0 else '#2196F3' for c in corrs]

    bars = ax.barh(range(len(names)), corrs, color=feat_colors, edgecolor='white')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Pearson r with pKd', fontweight='bold')
    ax.set_title('Feature Ranking\n(Correlation with Affinity)', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')

    fig.suptitle('Claim C: Satisfactory Interpretable Patterns',
                 fontweight='bold', fontsize=14)
    fig.savefig(os.path.join(FIG_DIR, 'fig4_affinity_patterns.png'), facecolor='white')
    plt.close()


def main():
    print("MMDCG-DTA Case Study — Visualization Suite")
    reps, results = load_data()
    cids = sorted(reps.keys())
    yt = np.array([reps[c]['true_pKd'] for c in cids])
    print(f"Loaded {len(cids)} complexes, pKd range: [{yt.min():.2f}, {yt.max():.2f}]")

    fig1_prediction_performance(reps)
    fig2_molecular_mechanics(reps)
    fig3_hil_edge_analysis(reps)
    fig4_affinity_patterns(reps)

    print(f"\nAll 4 figures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
