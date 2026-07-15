#!/usr/bin/env python3
"""Feature extraction from frozen MMDCG-DTA encoder + Ridge/MLP predictor.
Bypasses all DGL training — extracts GNN+physics features, then trains scikit-learn."""

import os, sys, pickle, time, json, numpy as np
import torch, dgl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))
from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"
RESULTS = f"{CASE_DIR}/extracted_features.pkl"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load data ──────────────────────────────────────────────────
for gp in ["hiv_protease_graphs_full.pkl", "hiv_protease_graphs.pkl"]:
    if os.path.exists(gp):
        with open(gp, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"Loaded {len(graph_data)} graphs from {gp}")
        break

all_ids = list(graph_data.keys())
labels = {cid: graph_data[cid]["label"] for cid in all_ids}
y_all = np.array([labels[c] for c in all_ids])
print(f"Labels: mean={y_all.mean():.3f}, std={y_all.std():.3f}, "
      f"range=[{y_all.min():.2f}, {y_all.max():.2f}], N={len(y_all)}")

# ── Load pretrained model (full, including head) ─────────────
config = {
    'embedding_dim': 64, 'd_atom': 4.0, 'd_res': 8.0, 'd_sub': 8.0,
    'l_intra': 2, 'l_inter': 2, 'l_atom': 2, 'l_sub': 2,
    'inter_negative_slope': 0.2,
    'sub_x_dim': 5, 'raw_atom_dim': 5, 'prot_res_dim': 1,
}
model = MMDCGDTAModel_Stage1(config).to(device)
state = torch.load("../Data/stage1_model_final.pth", map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()
print(f"Model loaded ({len(state)} pretrained keys)")

# ── Extract features using hooks ──────────────────────────────
features = {}  # {cid: {"gnn": array(64), "physics": array(9), "pred": float}}

# We'll capture intermediate features via forward hook
# The model's forward produces fusion_rep (285-dim) before pred_fc
# We'll save: lig_intra mean-pool, physics, and final prediction

fusion_features = {}
def get_fusion_hook(name):
    def hook(module, input, output):
        fusion_features[name] = output.detach().cpu().squeeze()
    return hook

# Register hook on pred_fc to get the fusion representation
handle = model.pred_fc.register_forward_hook(get_fusion_hook("fusion"))

physics_vals = {}
def get_physics_hook(name):
    def hook(module, input, output):
        physics_vals[name] = output.detach().cpu()
    return hook

# ── Run extraction ────────────────────────────────────────────
print(f"Extracting features for {len(all_ids)} complexes...")
t0 = time.time()
failed = 0
last_print = t0

# Process in batches for efficiency
for i, cid in enumerate(all_ids):
    if (i + 1) % 50 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / max(1, elapsed / 60)
        print(f"  [{i+1}/{len(all_ids)}] {elapsed:.0f}s elapsed, ~{rate:.0f} samples/min")

    sample = graph_data[cid]
    try:
        with torch.no_grad():
            sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                         for k, v in sample.items() if k != 'label'}
            y_pred = model(sample_dev)
            fusion_vec = fusion_features.get("fusion", None)

            if fusion_vec is None:
                failed += 1
                continue

            # Extract GNN features: lig_intra mean-pool
            lig_graph = sample_dev["ligand_atom_graph"]
            n_lig = lig_graph.num_nodes()

            # Get ligand atom features from intra encoder
            lig_raw = lig_graph.ndata["h"]
            lig_intra = model.ligand_atom_intra_encoder(lig_graph, lig_raw)
            lig_pooled = dgl.readout_nodes(lig_graph, lig_intra, op='mean')

            # Get physics: use the model's internal physics computation
            # Bond energies
            L_E_bond = torch.zeros(1, 1, device=device)
            if lig_graph.num_edges() > 0:
                src, dst = lig_graph.edges()
                pos = lig_graph.ndata['pos']
                dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
                L_E_bond = dgl.readout_edges(lig_graph, model.ligand_bond_sim(dist), op='mean')

            # Angle/Torsion
            L_E_angle, L_E_torsion = model._calc_angle_energy(lig_graph, model.ligand_angle_sim, lig_intra)

            # Protein bond
            prot_graph = sample_dev["protein_atom_graph"]
            P_E_bond = torch.zeros(1, 1, device=device)
            if prot_graph.num_edges() > 0:
                src, dst = prot_graph.edges()
                pos = prot_graph.ndata['pos']
                dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
                P_E_bond = dgl.readout_edges(prot_graph, model.protein_bond_sim(dist), op='mean')

            prot_intra = model.protein_atom_intra_encoder(prot_graph, prot_graph.ndata["h"])
            P_E_angle, P_E_torsion = model._calc_angle_energy(prot_graph, model.protein_angle_sim, prot_intra)

            # Inter-molecular energies
            inter_graph = sample_dev.get("atom_interaction_graph")
            I_vdw = torch.zeros(1, 1, device=device)
            I_elec = torch.zeros(1, 1, device=device)
            I_hbond = torch.zeros(1, 1, device=device)
            if inter_graph is not None and inter_graph.num_edges() > 0:
                I_vdw, I_elec, I_hbond = model._calc_inter_energy(inter_graph, lig_intra, prot_intra)

            physics = torch.cat([L_E_bond, L_E_angle, L_E_torsion,
                                P_E_bond, P_E_angle, P_E_torsion,
                                I_vdw, I_elec, I_hbond], dim=1)

            features[cid] = {
                "gnn": lig_pooled.cpu().numpy().flatten(),
                "physics": physics.cpu().numpy().flatten(),
                "fusion": fusion_vec.cpu().numpy().flatten(),
                "pred": y_pred.item(),
                "label": labels[cid],
            }

    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"  Error on {cid}: {type(e).__name__}: {str(e)[:100]}")

handle.remove()
elapsed = time.time() - t0
print(f"Extraction: {elapsed:.0f}s for {len(features)}/{len(all_ids)} complexes "
      f"({failed} failed)")

# ── Save features ─────────────────────────────────────────────
with open(RESULTS, "wb") as f:
    pickle.dump(features, f)
print(f"Features saved to {RESULTS}")

# ── Train predictors ──────────────────────────────────────────
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

cids_valid = list(features.keys())
y = np.array([features[c]["label"] for c in cids_valid])

# Feature sets
X_gnn = np.stack([features[c]["gnn"] for c in cids_valid])       # 64 dim
X_phys = np.stack([features[c]["physics"] for c in cids_valid])  # 9 dim
X_fusion = np.stack([features[c]["fusion"] for c in cids_valid]) # 285 dim
X_pred = np.array([features[c]["pred"] for c in cids_valid]).reshape(-1, 1)  # 1 dim
X_all = np.column_stack([X_gnn, X_phys])
X_all_fusion = np.column_stack([X_gnn, X_phys, X_fusion])

print(f"\nFeature dimensions:")
print(f"  GNN: {X_gnn.shape}, Physics: {X_phys.shape}")
print(f"  Fusion: {X_fusion.shape}")
print(f"  Combined (GNN+Phys): {X_all.shape}")

# Standardize
scaler = StandardScaler()
X_all_s = scaler.fit_transform(X_all)
X_fusion_s = StandardScaler().fit_transform(X_all_fusion)

# 5-fold CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(name, model, X, y, cv):
    y_pred = cross_val_predict(model, X, y, cv=cv)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    pearson = np.corrcoef(y, y_pred)[0, 1]
    print(f"  {name:20s} | RMSE={rmse:.4f}  MAE={mae:.4f}  Pearson={pearson:.4f}")
    return {"name": name, "RMSE": rmse, "MAE": mae, "Pearson": pearson}

print("\n=== Predictor Performance (5-fold CV, N=%d) ===" % len(y))
print("-" * 65)
results = []

# 1. Raw pretrained prediction
rmse_raw = np.sqrt(mean_squared_error(y, X_pred.flatten()))
mae_raw = mean_absolute_error(y, X_pred.flatten())
pearson_raw = np.corrcoef(y, X_pred.flatten())[0, 1]
print(f"  {'Pretrained raw pred':20s} | RMSE={rmse_raw:.4f}  MAE={mae_raw:.4f}  Pearson={pearson_raw:.4f}")
results.append({"name": "Pretrained raw pred", "RMSE": rmse_raw, "MAE": mae_raw, "Pearson": pearson_raw})

# 2. Ridge on GNN+Physics
results.append(evaluate("Ridge (GNN+Phys)", Ridge(alpha=1.0), X_all_s, y, cv))

# 3. Ridge with CV alpha
from sklearn.linear_model import RidgeCV
ridge_cv = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
results.append(evaluate("RidgeCV (GNN+Phys)", ridge_cv, X_all_s, y, cv))

# 4. Random Forest
results.append(evaluate("RF (GNN+Phys)", RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1), X_all_s, y, cv))

# 5. Gradient Boosting
results.append(evaluate("GBR (GNN+Phys)", GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42), X_all_s, y, cv))

# 6. MLP (small)
results.append(evaluate("MLP (GNN+Phys)", MLPRegressor(
    hidden_layer_sizes=(64, 32), activation='relu', alpha=0.01,
    max_iter=1000, early_stopping=True, random_state=42), X_all_s, y, cv))

# 7. Ridge on all fusion features
results.append(evaluate("Ridge (All Fusion)", Ridge(alpha=1.0), X_fusion_s, y, cv))

# ── Save results ──────────────────────────────────────────────
print(f"\n--- Best Result ---")
best = max(results, key=lambda r: r["Pearson"])
print(f"Best: {best['name']} — RMSE={best['RMSE']:.4f}, Pearson={best['Pearson']:.4f}")

target_rmse = 0.59
target_pearson = 0.85
print(f"\nTarget: RMSE < {target_rmse}, Pearson > {target_pearson}")
if best["RMSE"] <= target_rmse and best["Pearson"] >= target_pearson:
    print("TARGET MET!")
else:
    gap_rmse = best["RMSE"] - target_rmse
    gap_p = target_pearson - best["Pearson"]
    print(f"Gap: RMSE +{gap_rmse:.3f}, Pearson -{gap_p:.3f}")

with open(f"{CASE_DIR}/predictor_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {CASE_DIR}/predictor_results.json")
