"""Fix the flat clustered heatmap with better aspect ratio and readability."""
import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy import stats
import warnings, os
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 13,
    'axes.titlesize': 16, 'axes.labelsize': 14,
    'figure.dpi': 200, 'savefig.dpi': 200,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.15,
})

C_HIGH, C_PROTEIN = '#E63946', '#457B9D'
os.makedirs('./figures', exist_ok=True)

print("Loading multi-stage report data...")
with open('multistage_case_study_report.json', 'r') as f:
    report = json.load(f)

# Use Stage 1 per_complex data (has interpretability fields)
per_complex = report['per_complex']
comp_ids = sorted(per_complex.keys())

# Build feature matrix
true_pkd = np.array([per_complex[c]['true_pKd'] for c in comp_ids])
sort_idx = np.argsort(true_pkd)
comp_ids_sorted = [comp_ids[i] for i in sort_idx]
true_sorted = true_pkd[sort_idx]

interp_keys = sorted([k for k in per_complex[comp_ids[0]]['interpretability'].keys()
                      if k != 'complex_id'])

feature_matrix = []
feature_names = []
for k in interp_keys:
    vals = np.array([per_complex[c]['interpretability'].get(k, np.nan) for c in comp_ids_sorted])
    if np.std(vals[~np.isnan(vals)]) > 1e-8:
        feature_matrix.append(np.nan_to_num(vals, nan=0.0))
        # Clean feature names for readability
        name = k.replace('atom_', '').replace('_', ' ').replace('I ', 'Inter ')
        name = name.replace('ligand', 'Lig').replace('protein', 'Prot')
        name = name.replace('intra', 'intra').replace('inter', 'inter')
        name = name.replace('hil change', 'HIL')
        name = name.replace('energy', 'E')
        feature_names.append(name)

feature_matrix = np.array(feature_matrix).T
feature_z = (feature_matrix - feature_matrix.mean(axis=0)) / (feature_matrix.std(axis=0) + 1e-8)

# Compute correlations
corrs = []
for j in range(feature_matrix.shape[1]):
    r_val, _ = stats.pearsonr(feature_matrix[:, j], true_sorted)
    corrs.append(r_val)

# Clustering
row_linkage = hierarchy.linkage(feature_z, method='ward')
col_linkage = hierarchy.linkage(feature_z.T, method='ward')

# ============================================================
# FIGURE: Better-proportioned clustered heatmap
# ============================================================
# Taller figure with better aspect ratio - fix the "flat" problem
fig = plt.figure(figsize=(20, 16))

gs = fig.add_gridspec(2, 2, height_ratios=[20, 1], width_ratios=[1, 12],
                      hspace=0.02, wspace=0.02)

ax_heat = fig.add_subplot(gs[0, 1])
ax_row = fig.add_subplot(gs[0, 0], sharey=ax_heat)
ax_col = fig.add_subplot(gs[1, 1], sharex=ax_heat)

# Row dendrogram
row_dendro = hierarchy.dendrogram(row_linkage, orientation='left', ax=ax_row,
                                   no_labels=True, color_threshold=0.5*max(row_linkage[:, 2]),
                                   above_threshold_color='#888')
ax_row.axis('off')
row_order = row_dendro['leaves']

# Column dendrogram
col_dendro = hierarchy.dendrogram(col_linkage, ax=ax_col,
                                   no_labels=True, color_threshold=0.5*max(col_linkage[:, 2]),
                                   above_threshold_color='#888')
ax_col.axis('off')
col_order = col_dendro['leaves']

# Reorder
feature_z_ord = feature_z[row_order, :][:, col_order]
feature_names_ord = [feature_names[i] for i in col_order]
corrs_ord = [corrs[i] for i in col_order]
true_ord = true_sorted[row_order]

# Heatmap - larger, more readable
im = ax_heat.imshow(feature_z_ord, aspect='auto', cmap='RdBu_r',
                    interpolation='bilinear', vmin=-2.5, vmax=2.5)

# Feature names - larger font, angled for readability
ax_heat.set_xticks(range(len(feature_names_ord)))
ax_heat.set_xticklabels(feature_names_ord, rotation=45, ha='right', fontsize=9)
ax_heat.set_yticks([])
ax_heat.set_xlim(-0.5, len(feature_names_ord) - 0.5)

# Correlation annotation at top
for j in range(len(feature_names_ord)):
    color = C_HIGH if corrs_ord[j] > 0 else C_PROTEIN
    ax_heat.text(j, -1.8, f'{corrs_ord[j]:+.2f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color=color, rotation=45)

# pKd overlay on right side
ax_pkd = ax_heat.twinx()
ax_pkd.set_ylim(ax_heat.get_ylim())
ax_pkd.plot(true_ord, np.arange(len(true_ord)), '-', color='black',
            linewidth=2.5, alpha=0.7, label='True pKd')
ax_pkd.set_ylabel('Experimental pKd', fontweight='bold', fontsize=13, rotation=270, labelpad=15)
ax_pkd.legend(fontsize=11, loc='lower right')
ax_pkd.set_yticks([])

# Colorbar - properly positioned
cbar_ax = fig.add_axes([0.93, 0.55, 0.012, 0.38])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label('Z-score', fontsize=12)

ax_heat.set_title('Hierarchical Clustering of Interpretability Features Across 50 HIV-1 PR Complexes',
                  fontweight='bold', fontsize=15, pad=15)

fig.savefig('./figures/fig13_clustered_heatmap.png', facecolor='white')
plt.close()
print("Saved fixed fig13_clustered_heatmap.png (20x16, readable labels)")

# ============================================================
# ALSO: Feature correlation bar chart with English labels
# ============================================================
fig, ax = plt.subplots(figsize=(10, 7))

corr_pairs = []
for j, name in enumerate(feature_names):
    r_val = corrs[j]
    corr_pairs.append((abs(r_val), r_val, name))

corr_pairs.sort(reverse=True)
top_n = 14
top_features = corr_pairs[:top_n]

feature_labels = [d[2] for d in top_features]
feature_corrs = [d[1] for d in top_features]
feature_colors = [C_HIGH if c > 0 else C_PROTEIN for c in feature_corrs]

bars = ax.barh(range(len(feature_labels)), feature_corrs, color=feature_colors,
               edgecolor='white', height=0.65)

ax.set_yticks(range(len(feature_labels)))
ax.set_yticklabels(feature_labels, fontsize=12)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Pearson r with Experimental pKd', fontweight='bold')
ax.set_title('Top Interpretability Features Correlated with Binding Affinity',
             fontweight='bold', pad=12)
ax.grid(True, alpha=0.15, axis='x')

for i, (bar, d) in enumerate(zip(bars, top_features)):
    direction = 0.02 * np.sign(d[1])
    ax.text(d[1] + direction, i, f'{d[1]:+.3f}',
            va='center', fontsize=10, fontweight='bold')

ax.set_xlim(-0.7, 0.7)

fig.savefig('./figures/fig11_feature_correlation.png', facecolor='white')
plt.close()
print("Saved fig11_feature_correlation.png (English labels)")

print("\nDone! Fixed figures saved.")
