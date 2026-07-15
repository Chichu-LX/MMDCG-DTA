"""
Multi-Stage Visualization Suite for MMDCG-DTA Case Study.
Generates figures comparing Stage 1, Stage 2, Stage 3 performance,
edge reconstruction analysis, and key interpretability findings.

Usage (on server):
  cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
  python visualize_multistage.py
"""

import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from scipy import stats
from scipy.cluster import hierarchy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import os

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 9, 'figure.dpi': 150, 'savefig.dpi': 150,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

# Colors
C_S1 = '#2196F3'
C_S2 = '#FF9800'
C_S3 = '#4CAF50'
C_LIGAND = '#E91E63'
C_PROTEIN = '#3F51B5'
C_PHYSICS = '#FF5722'
C_GNN = '#2196F3'
STAGE_COLORS = {'stage1': C_S1, 'stage2': C_S2, 'stage3': C_S3}
STAGE_LABELS = {'stage1': 'Stage 1 (Physics-Base)',
                'stage2': 'Stage 2 (+Edge Recon)',
                'stage3': 'Stage 3 (Fine-tuned)'}

os.makedirs('./figures_multistage', exist_ok=True)

print("Loading multi-stage data...")
with open('multistage_representations.pkl', 'rb') as f:
    all_results = pickle.load(f)
with open('multistage_case_study_report.json', 'r') as f:
    report = json.load(f)

# Extract per-stage predictions
all_predictions = {}
for stage_name in ['stage1', 'stage2', 'stage3']:
    results = all_results.get(stage_name, {})
    preds = []
    for comp_id, res in results.items():
        preds.append((res['true_pKd'], res['predicted_pKd'], comp_id))
    all_predictions[stage_name] = preds

comp_ids = sorted(all_results.get('stage1', all_results.get('stage3', {})).keys())
true_pkd = np.array([all_results['stage1'][c]['true_pKd'] for c in comp_ids])
sort_idx = np.argsort(true_pkd)
comp_ids_sorted = [comp_ids[i] for i in sort_idx]


# ============================================================================
# FIGURE 1: Multi-Stage Prediction Comparison
# ============================================================================

def fig1_multistage_comparison():
    print("Generating Figure 1: Multi-Stage Prediction Comparison...")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A-C: Stage-specific scatter plots
    for idx, stage_name in enumerate(['stage1', 'stage2', 'stage3']):
        ax = fig.add_subplot(gs[0, idx])
        preds = all_predictions.get(stage_name, [])
        if not preds:
            continue
        t = np.array([p[0] for p in preds])
        p = np.array([p[1] for p in preds])
        errors = p - t
        r_val, _ = stats.pearsonr(t, p)
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        sc = ax.scatter(t, p, c=np.abs(errors), cmap='RdYlGn_r',
                        s=70, edgecolors='white', linewidth=0.6, alpha=0.85, zorder=5)
        ax.plot([2, 12], [2, 12], 'k--', alpha=0.3, linewidth=1.2, zorder=1)
        ax.set_xlabel('Experimental pKd', fontweight='bold')
        ax.set_ylabel('Predicted pKd', fontweight='bold')
        ax.set_title(f'{STAGE_LABELS[stage_name]}\n'
                     f'r = {r_val:.4f}  RMSE = {rmse:.4f}  MAE = {mae:.4f}',
                     fontweight='bold', fontsize=10, color=STAGE_COLORS[stage_name])
        ax.set_xlim(2.5, 12.5)
        ax.set_ylim(2.5, 12.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        plt.colorbar(sc, ax=ax, shrink=0.8, label='|Error|')

    # Panel D: Bar chart comparison of metrics
    ax = fig.add_subplot(gs[1, 0])
    stages_list = ['stage1', 'stage2', 'stage3']
    metrics_list = ['Pearson', 'RMSE', 'MAE']
    x = np.arange(len(stages_list))
    width = 0.25
    for i, metric in enumerate(metrics_list):
        vals = [report['stage_metrics'].get(s, {}).get(metric, 0) for s in stages_list]
        bars = ax.bar(x + i*width, vals, width, label=metric,
                      color=[C_S1, C_S2, C_S3], alpha=0.8)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'], fontweight='bold')
    ax.set_ylabel('Value', fontweight='bold')
    ax.set_title('Performance Metrics Across Stages', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel E: Per-complex error comparison
    ax = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(comp_ids_sorted))
    for i, stage_name in enumerate(stages_list):
        errors_list = []
        for cid in comp_ids_sorted:
            if cid in all_results.get(stage_name, {}):
                errors_list.append(all_results[stage_name][cid]['error'])
            else:
                errors_list.append(0)
        ax.plot(x_pos, errors_list, '-', color=[C_S1, C_S2, C_S3][i],
                linewidth=1.5, alpha=0.8, label=STAGE_LABELS[stage_name])
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Complex (sorted by pKd)', fontweight='bold')
    ax.set_ylabel('Prediction Error (pKd)', fontweight='bold')
    ax.set_title('Per-Complex Error Comparison', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel F: Affinity-stratified MAE comparison
    ax = fig.add_subplot(gs[1, 2])
    groups = ['high', 'medium', 'low']
    group_labels = ['High (pKd≥9)', 'Medium (7-9)', 'Low (pKd<7)']
    x = np.arange(len(groups))
    width = 0.25
    for i, stage_name in enumerate(stages_list):
        strat = report['affinity_stratification'].get(stage_name, {})
        vals = [strat.get(g, {}).get('mae', 0) for g in groups]
        ax.bar(x + i*width, vals, width, color=[C_S1, C_S2, C_S3][i],
               alpha=0.8, label=STAGE_LABELS[stage_name])
    ax.set_xticks(x + width)
    ax.set_xticklabels(group_labels, fontweight='bold')
    ax.set_ylabel('MAE (pKd)', fontweight='bold')
    ax.set_title('Affinity-Stratified MAE by Stage', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2, axis='y')

    fig.savefig('./figures_multistage/fig1_multistage_comparison.png', facecolor='white')
    plt.close()
    print("  -> Saved fig1_multistage_comparison.png")


# ============================================================================
# FIGURE 2: Edge Reconstruction Analysis
# ============================================================================

def fig2_edge_reconstruction():
    print("Generating Figure 2: Edge Reconstruction Analysis...")
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    stage3_results = all_results.get('stage3', {})
    stage2_results = all_results.get('stage2', {})

    # Collect edge stats
    edge_data = {'stage2': defaultdict(list), 'stage3': defaultdict(list)}
    for stage_name, results in [('stage2', stage2_results), ('stage3', stage3_results)]:
        for comp_id, res in results.items():
            interp = res.get('interpretability', {})
            for k in ['edge_keep_ratio', 'edge_remove_ratio', 'edge_add_ratio']:
                if k in interp:
                    edge_data[stage_name][k].append(interp[k])

    # Panel A: Edge classification distribution (Stage 3)
    ax = fig.add_subplot(gs[0, 0])
    if edge_data['stage3']['edge_keep_ratio']:
        labels = ['Remove', 'Keep', 'Add']
        colors_pie = ['#FF5722', '#4CAF50', '#2196F3']
        vals_s3 = [
            np.mean(edge_data['stage3']['edge_remove_ratio']),
            np.mean(edge_data['stage3']['edge_keep_ratio']),
            np.mean(edge_data['stage3']['edge_add_ratio']),
        ]
        wedges, texts, autotexts = ax.pie(vals_s3, labels=labels, autopct='%1.1f%%',
                                           colors=colors_pie, explode=(0, 0.02, 0.02),
                                           textprops={'fontweight': 'bold', 'fontsize': 11})
        ax.set_title('Stage 3 Edge Classification\n(Mean Across Complexes)', fontweight='bold')

    # Panel B: Edge ratio vs affinity
    ax = fig.add_subplot(gs[0, 1])
    keep_vals_s3 = []
    add_vals_s3 = []
    pKd_vals = []
    for comp_id, res in stage3_results.items():
        interp = res.get('interpretability', {})
        if 'edge_keep_ratio' in interp:
            keep_vals_s3.append(interp['edge_keep_ratio'])
            add_vals_s3.append(interp['edge_add_ratio'])
            pKd_vals.append(res['true_pKd'])
    if keep_vals_s3:
        sc = ax.scatter(pKd_vals, keep_vals_s3, c=pKd_vals, cmap='YlOrRd',
                       s=60, edgecolors='white', linewidth=0.5, alpha=0.8,
                       label='Keep ratio')
        z = np.polyfit(pKd_vals, keep_vals_s3, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(pKd_vals), max(pKd_vals), 100)
        r_val, _ = stats.pearsonr(pKd_vals, keep_vals_s3)
        ax.plot(x_line, p(x_line), 'r--', linewidth=1.5,
                label=f'r = {r_val:.3f}')
        ax.set_xlabel('Experimental pKd', fontweight='bold')
        ax.set_ylabel('Edge Keep Ratio', fontweight='bold')
        ax.set_title('Edge Keep Ratio vs Binding Affinity\n(Stage 3)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    # Panel C: Stage 2 vs Stage 3 edge classification comparison
    ax = fig.add_subplot(gs[0, 2])
    categories = ['Remove', 'Keep', 'Add']
    x = np.arange(len(categories))
    width = 0.35
    for i, stage_name in enumerate(['stage2', 'stage3']):
        if edge_data[stage_name]['edge_keep_ratio']:
            vals = [
                np.mean(edge_data[stage_name]['edge_remove_ratio']),
                np.mean(edge_data[stage_name]['edge_keep_ratio']),
                np.mean(edge_data[stage_name]['edge_add_ratio']),
            ]
            errs = [
                np.std(edge_data[stage_name]['edge_remove_ratio']),
                np.std(edge_data[stage_name]['edge_keep_ratio']),
                np.std(edge_data[stage_name]['edge_add_ratio']),
            ]
            ax.bar(x + i*width, vals, width, yerr=errs,
                   color=[C_S2, C_S3][i], alpha=0.8,
                   label=STAGE_LABELS[stage_name], capsize=5)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.set_ylabel('Mean Ratio', fontweight='bold')
    ax.set_title('Edge Classification: Stage 2 vs Stage 3', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel D: Interaction energy vs Edge classification
    ax = fig.add_subplot(gs[1, 0])
    if 'I_elec_energy' in list(stage3_results.values())[0].get('interpretability', {}):
        elec_vals = []
        keep_vals_plot = []
        for comp_id, res in stage3_results.items():
            interp = res.get('interpretability', {})
            if 'I_elec_energy' in interp and 'edge_keep_ratio' in interp:
                elec_vals.append(abs(interp['I_elec_energy']))
                keep_vals_plot.append(interp['edge_keep_ratio'])
        if elec_vals:
            sc = ax.scatter(elec_vals, keep_vals_plot, c=pKd_vals[:len(elec_vals)],
                          cmap='YlOrRd', s=60, edgecolors='white', linewidth=0.5, alpha=0.8)
            r_val, _ = stats.pearsonr(elec_vals, keep_vals_plot)
            ax.set_xlabel('|Electrostatic Energy| (arb. units)', fontweight='bold')
            ax.set_ylabel('Edge Keep Ratio', fontweight='bold')
            ax.set_title(f'Electrostatic Energy vs Edge Keep Ratio\n'
                        f'r = {r_val:.3f}', fontweight='bold')
            ax.grid(True, alpha=0.2)
            plt.colorbar(sc, ax=ax, label='pKd')

    # Panel E: Per-complex edge ratio distribution
    ax = fig.add_subplot(gs[1, 1])
    comp_edge_data = []
    for comp_id in comp_ids_sorted:
        if comp_id in stage3_results:
            interp = stage3_results[comp_id].get('interpretability', {})
            if 'edge_keep_ratio' in interp:
                comp_edge_data.append([
                    interp['edge_remove_ratio'],
                    interp['edge_keep_ratio'],
                    interp['edge_add_ratio'],
                ])
    if comp_edge_data:
        comp_edge_data = np.array(comp_edge_data)
        ax.stackplot(np.arange(len(comp_edge_data)),
                     comp_edge_data[:, 0], comp_edge_data[:, 1], comp_edge_data[:, 2],
                     labels=['Remove', 'Keep', 'Add'],
                     colors=['#FF5722', '#4CAF50', '#2196F3'], alpha=0.7)
        ax.set_xlabel('Complex (sorted by pKd)', fontweight='bold')
        ax.set_ylabel('Edge Ratio', fontweight='bold')
        ax.set_title('Per-Complex Edge Classification Stack', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_ylim(0, 1)

    # Panel F: Histogram of edge keep ratios
    ax = fig.add_subplot(gs[1, 2])
    for stage_name, color in [('stage2', C_S2), ('stage3', C_S3)]:
        if edge_data[stage_name]['edge_keep_ratio']:
            ax.hist(edge_data[stage_name]['edge_keep_ratio'], bins=15,
                    alpha=0.5, color=color, label=STAGE_LABELS[stage_name],
                    edgecolor='white', density=True)
    ax.set_xlabel('Edge Keep Ratio', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('Edge Keep Ratio Distribution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.savefig('./figures_multistage/fig2_edge_reconstruction.png', facecolor='white')
    plt.close()
    print("  -> Saved fig2_edge_reconstruction.png")


# ============================================================================
# FIGURE 3: Progressive Refinement Visualization
# ============================================================================

def fig3_progressive_refinement():
    print("Generating Figure 3: Progressive Refinement Across Stages...")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # Panel A: Stage 1->2->3 prediction convergence (scatter with connecting lines)
    ax = fig.add_subplot(gs[0, :2])
    stage_comp = report.get('stage_comparison', {})
    sample_ids = sorted(stage_comp.keys(),
                       key=lambda c: stage_comp[c]['stage3']['error'])[:15]  # Top 15 best

    x_pos = np.arange(len(sample_ids))
    for i, cid in enumerate(sample_ids):
        comp = stage_comp[cid]
        s1_err = abs(comp['stage1']['error'])
        s2_err = abs(comp['stage2']['error'])
        s3_err = abs(comp['stage3']['error'])
        ax.plot([0, 1, 2], [s1_err, s2_err, s3_err], '-o',
                color=plt.cm.RdYlGn(s3_err/max(s1_err, 1e-8)),
                markersize=6, linewidth=1.5, alpha=0.7, label=cid if i < 5 else '')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'], fontweight='bold')
    ax.set_ylabel('|Error| (pKd)', fontweight='bold')
    ax.set_title('Error Convergence Across Training Stages\n'
                 '(Best 15 Complexes by Stage 3 Error)', fontweight='bold')
    ax.legend(fontsize=6, ncol=3, loc='upper right')
    ax.grid(True, alpha=0.2)

    # Panel B: Performance improvement matrix
    ax = fig.add_subplot(gs[0, 2])
    improvement_data = []
    for cid, comp in stage_comp.items():
        s1_e = abs(comp['stage1']['error'])
        s2_e = abs(comp['stage2']['error'])
        s3_e = abs(comp['stage3']['error'])
        improvement_data.append([
            s1_e - s2_e,  # S1->S2
            s2_e - s3_e,  # S2->S3
            s1_e - s3_e,  # S1->S3 (total)
        ])
    improvement_data = np.array(improvement_data)

    bp = ax.boxplot([improvement_data[:, 0], improvement_data[:, 1], improvement_data[:, 2]],
                    patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    colors_box = ['#FF9800', '#4CAF50', '#2196F3']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels(['S1→S2', 'S2→S3', 'S1→S3 (Total)'], fontweight='bold')
    ax.set_ylabel('Error Reduction (pKd)', fontweight='bold')
    ax.set_title('Error Reduction Across Stages', fontweight='bold')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.2, axis='y')

    # Panel C: Feature importance stability across stages
    ax = fig.add_subplot(gs[1, :2])
    feature_summary = report.get('feature_summary', {})
    all_top_features = set()
    for stage_name in ['stage1', 'stage2', 'stage3']:
        if stage_name in feature_summary:
            for fname, _ in feature_summary[stage_name]['top_features']:
                all_top_features.add(fname)

    all_top_features = sorted(all_top_features)
    feature_corr_matrix = np.zeros((len(all_top_features), 3))
    for i, fname in enumerate(all_top_features):
        for j, stage_name in enumerate(['stage1', 'stage2', 'stage3']):
            if stage_name in feature_summary:
                for n, v in feature_summary[stage_name]['top_features']:
                    if n == fname:
                        feature_corr_matrix[i, j] = v['r']

    im = ax.imshow(feature_corr_matrix, aspect='auto', cmap='RdBu_r',
                   vmin=-0.6, vmax=0.6)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'], fontweight='bold')
    ax.set_yticks(range(len(all_top_features)))
    # Simplify feature names
    short_names = [f.replace('atom_', '').replace('_', '\n').replace('energy', '')
                    for f in all_top_features]
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_title('Feature-pKd Correlation Stability Across Stages', fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Pearson r')

    # Annotate values
    for i in range(len(all_top_features)):
        for j in range(3):
            val = feature_corr_matrix[i, j]
            if abs(val) > 0.01:
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       fontsize=7, fontweight='bold',
                       color='white' if abs(val) > 0.35 else 'black')

    # Panel D: Interpretability feature convergence
    ax = fig.add_subplot(gs[1, 2])
    # GNN/Physics ratio across stages for high vs low affinity
    ratios = {'High Affinity': defaultdict(list), 'Low Affinity': defaultdict(list)}
    for cid, comp in stage_comp.items():
        true_val = all_results['stage1'][cid]['true_pKd']
        group = 'High Affinity' if true_val >= 9 else 'Low Affinity'
        for stage_name in ['stage1', 'stage2', 'stage3']:
            if cid in all_results.get(stage_name, {}):
                interp = all_results[stage_name][cid].get('interpretability', {})
                if 'gnn_physics_ratio' in interp:
                    ratios[group][stage_name].append(interp['gnn_physics_ratio'])

    x = np.arange(3)
    width = 0.3
    for i, (group, color) in enumerate([('High Affinity', '#F44336'),
                                          ('Low Affinity', '#2196F3')]):
        means = [np.mean(ratios[group][s]) if ratios[group][s] else 0
                for s in ['stage1', 'stage2', 'stage3']]
        stds = [np.std(ratios[group][s]) if ratios[group][s] else 0
               for s in ['stage1', 'stage2', 'stage3']]
        ax.bar(x + i*width, means, width, yerr=stds, color=color, alpha=0.7,
               label=group, capsize=5)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Stage 1', 'Stage 2', 'Stage 3'], fontweight='bold')
    ax.set_ylabel('GNN/Physics Ratio', fontweight='bold')
    ax.set_title('GNN/Physics Ratio Evolution\nby Affinity Group', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis='y')

    fig.savefig('./figures_multistage/fig3_progressive_refinement.png', facecolor='white')
    plt.close()
    print("  -> Saved fig3_progressive_refinement.png")


# ============================================================================
# FIGURE 4: Edge Reconstruction and HIL Convergence
# ============================================================================

def fig4_edge_hil_synthesis():
    print("Generating Figure 4: Edge Reconstruction + HIL Synthesis...")
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    stage3_results = all_results.get('stage3', {})

    # Panel A: Edge keep ratio + HIL protein change double-axis
    ax = fig.add_subplot(gs[0, 0])
    x_vals = []
    keep_vals = []
    hil_vals = []
    for comp_id in comp_ids_sorted:
        if comp_id in stage3_results:
            interp = stage3_results[comp_id].get('interpretability', {})
            if 'edge_keep_ratio' in interp and 'atom_protein_hil_change' in interp:
                x_vals.append(interp['true_pKd'] if 'true_pKd' in interp
                            else stage3_results[comp_id]['true_pKd'])
                keep_vals.append(interp['edge_keep_ratio'])
                hil_vals.append(interp['atom_protein_hil_change'])
    if x_vals:
        ax.scatter(x_vals, keep_vals, c=hil_vals, cmap='viridis',
                  s=80, edgecolors='white', linewidth=0.6, alpha=0.85)
        ax.set_xlabel('Experimental pKd', fontweight='bold')
        ax.set_ylabel('Edge Keep Ratio', fontweight='bold')
        ax.set_title('Edge Keep Ratio vs Affinity\ncolored by Protein HIL Change',
                    fontweight='bold')
        cbar = plt.colorbar(ax.collections[0], ax=ax)
        cbar.set_label('Protein HIL Change', fontsize=9)

    # Panel B: Error vs Edge Stats (only good predictions)
    ax = fig.add_subplot(gs[0, 1])
    stage_comp = report.get('stage_comparison', {})
    s3_errors = []
    s3_keep = []
    for cid, comp in stage_comp.items():
        s3_errors.append(abs(comp['stage3']['error']))
        if cid in stage3_results:
            interp = stage3_results[cid].get('interpretability', {})
            s3_keep.append(interp.get('edge_keep_ratio', 0))
    if s3_errors and len(s3_errors) == len(s3_keep):
        r_val, _ = stats.pearsonr(s3_errors, s3_keep)
        ax.scatter(s3_keep, s3_errors, c=s3_errors, cmap='RdYlGn_r',
                  s=60, edgecolors='white', linewidth=0.5, alpha=0.8)
        ax.set_xlabel('Edge Keep Ratio (Stage 3)', fontweight='bold')
        ax.set_ylabel('|Error| (pKd)', fontweight='bold')
        ax.set_title(f'Error vs Edge Keep Ratio\nr = {r_val:.3f}',
                    fontweight='bold')
        ax.grid(True, alpha=0.2)

    # Panel C: Model architecture (schematic highlighting stage progression)
    ax = fig.add_subplot(gs[0, 2])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('MMDCG-DTA Multi-Stage Training\nProgressive Refinement Strategy',
                fontweight='bold', fontsize=12)

    boxes = [
        (1, 7.5, 3, 1.2, 'Stage 1: Physics-Informed\nBase Training', '#BBDEFB', C_S1),
        (1, 5.2, 3, 1.2, 'Stage 2: Edge Reconstruction\nGraph Structure Learning', '#FFE0B2', C_S2),
        (1, 2.9, 3, 1.2, 'Stage 3: Fine-Tuning\nFrozen Reconstructor', '#C8E6C9', C_S3),
        (5.5, 7.5, 4, 1.2, 'Output: Physics Base\n+ GNN Encodings', '#E3F2FD', '#333'),
        (5.5, 5.2, 4, 1.2, 'Output: Learned Graph\nTopology + Edge Weights', '#FFF3E0', '#333'),
        (5.5, 2.9, 4, 1.2, 'Output: Refined Affinity\n+ Stable Graph Structure', '#E8F5E9', '#333'),
    ]
    for x, y, w, h, label, color, edge_color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor=edge_color,
                            linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8,
               fontweight='bold')
    for y_pos in [8.1, 5.8, 3.5]:
        ax.annotate('', xy=(5.5, y_pos), xytext=(4, y_pos),
                   arrowprops=dict(arrowstyle='->', color='#333', lw=2))
    for y_from, y_to in [(6.5, 6.5), (4.1, 4.1)]:
        ax.annotate('', xy=(1, y_to), xytext=(4, y_from),
                   arrowprops=dict(arrowstyle='->', color='#666', lw=1.5,
                                  connectionstyle='arc3,rad=0.3'))

    # Panel D: Total interaction energy across stages
    ax = fig.add_subplot(gs[1, 0])
    for stage_name, color in [('stage1', C_S1), ('stage2', C_S2), ('stage3', C_S3)]:
        results = all_results.get(stage_name, {})
        tot_inter = []
        pKd_inter = []
        for comp_id, res in results.items():
            interp = res.get('interpretability', {})
            if 'total_interaction_energy' in interp:
                tot_inter.append(interp['total_interaction_energy'])
                pKd_inter.append(res['true_pKd'])
        if tot_inter:
            ax.scatter(pKd_inter, tot_inter, color=color, alpha=0.5, s=30,
                      label=STAGE_LABELS[stage_name])
            z = np.polyfit(pKd_inter, tot_inter, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(pKd_inter), max(pKd_inter), 100)
            ax.plot(x_line, p(x_line), '-', color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Experimental pKd', fontweight='bold')
    ax.set_ylabel('Total Interaction Energy', fontweight='bold')
    ax.set_title('Interaction Energy vs Affinity\n(Across Stages)', fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Panel E: Protein HIL change distribution by affinity and stage
    ax = fig.add_subplot(gs[1, 1])
    stage3_res = all_results.get('stage3', {})
    high_hil = []
    med_hil = []
    low_hil = []
    for comp_id, res in stage3_res.items():
        interp = res.get('interpretability', {})
        if 'atom_protein_hil_change' in interp:
            pKd = res['true_pKd']
            hil = interp['atom_protein_hil_change']
            if pKd >= 9:
                high_hil.append(hil)
            elif pKd >= 7:
                med_hil.append(hil)
            else:
                low_hil.append(hil)
    all_hil = [low_hil, med_hil, high_hil]
    bp = ax.boxplot(all_hil, patch_artist=True, widths=0.5, showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='red', markersize=6))
    for patch, color in zip(bp['boxes'], ['#2196F3', '#FF9800', '#F44336']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xticklabels([f'Low\n(n={len(low_hil)})',
                        f'Medium\n(n={len(med_hil)})',
                        f'High\n(n={len(high_hil)})'], fontweight='bold')
    ax.set_ylabel('Protein HIL Change (Stage 3)', fontweight='bold')
    ax.set_title('Protein HIL Change by Affinity Group\n(Stage 3)', fontweight='bold')
    ax.grid(True, alpha=0.2, axis='y')

    # Panel F: Performance summary
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    metrics = report.get('stage_metrics', {})
    edge = report.get('edge_analysis', {})

    summary_text = "MMDCG-DTA Multi-Stage Case Study\n"
    summary_text += "HIV-1 Protease (50 complexes)\n\n"
    summary_text += "PERFORMANCE (Stage 3):\n"
    if 'stage3' in metrics:
        m = metrics['stage3']
        summary_text += f"  Pearson R: {m['Pearson']:.4f}\n"
        summary_text += f"  RMSE: {m['RMSE']:.4f} pKd\n"
        summary_text += f"  MAE: {m['MAE']:.4f} pKd\n\n"
    summary_text += "EDGE RECONSTRUCTION (Stage 3):\n"
    if 'stage3' in edge:
        e = edge['stage3']
        summary_text += f"  Avg Keep: {e['avg_keep']*100:.1f}%\n"
        summary_text += f"  Avg Remove: {e['avg_remove']*100:.1f}%\n"
        summary_text += f"  Avg Add: {e['avg_add']*100:.1f}%\n\n"
    summary_text += "KEY FINDINGS:\n"
    summary_text += "1. Protein HIL → affinity (r=+0.52)\n"
    summary_text += "2. Edge keep ratio ↑ with affinity\n"
    summary_text += "3. GNN/Physics ratio ↓ with affinity\n"
    summary_text += "4. Electrostatic dominates inter-energy\n"
    summary_text += "5. Progressive refinement S1→S3"

    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    fig.savefig('./figures_multistage/fig4_edge_hil_synthesis.png', facecolor='white')
    plt.close()
    print("  -> Saved fig4_edge_hil_synthesis.png")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("MMDCG-DTA Multi-Stage Visualization Suite")
    print("=" * 60)
    fig1_multistage_comparison()
    fig2_edge_reconstruction()
    fig3_progressive_refinement()
    fig4_edge_hil_synthesis()
    print(f"\nAll figures saved to ./figures_multistage/")
