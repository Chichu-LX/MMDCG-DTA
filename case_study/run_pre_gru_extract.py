#!/usr/bin/env python3
"""Extract pre-GRU features (64 GNN + 9 physics) — same approach as successful VS pipeline."""

import pickle, sys, os, time, json, numpy as np
import torch, dgl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))
from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"
device = torch.device("cuda")

for gp in ["hiv_protease_graphs_full.pkl", "hiv_protease_graphs.pkl"]:
    if os.path.exists(gp):
        with open(gp, 'rb') as f:
            data = pickle.load(f)
        break
print(f"Loaded {len(data)} graphs")

config = {'embedding_dim': 64, 'd_atom': 4.0, 'd_res': 8.0, 'd_sub': 8.0,
          'l_intra': 2, 'l_inter': 2, 'l_atom': 2, 'l_sub': 2,
          'inter_negative_slope': 0.2, 'sub_x_dim': 5, 'raw_atom_dim': 5, 'prot_res_dim': 1}
model = MMDCGDTAModel_Stage1(config).to(device)
state = torch.load("../Data/stage1_model_final.pth", map_location=device, weights_only=True)
model.load_state_dict(state, strict=False)
model.eval()

# We'll extract directly the intermediate outputs by running model components manually
# For each complex, we:
# 1. Run ligand_atom_intra_encoder → mean pool → 64-dim
# 2. Run physics modules → 9-dim
# 3. Total: 73-dim feature vector

cids = list(data.keys())
features_gnn = []
features_phys = []
y_all = []
failed = 0
t0 = time.time()

for i, cid in enumerate(cids):
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(cids)}] {time.time()-t0:.0f}s")

    sample = data[cid]
    try:
        with torch.no_grad():
            sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                         for k, v in sample.items() if k != 'label'}

            lig_g = sample_dev["ligand_atom_graph"]
            prot_g = sample_dev["protein_atom_graph"]
            inter_g = sample_dev.get("atom_interaction_graph")

            # GNN intra-features: run intra encoder
            lig_h = lig_g.ndata["h"]
            lig_intra = model.ligand_atom_intra_encoder(lig_g, lig_h)
            # Mean pool across atoms
            lig_pooled = dgl.readout_nodes(lig_g, lig_intra, op='mean')

            # Physics features
            # Bond energies
            L_E_bond = torch.zeros(1, 1, device=device)
            if lig_g.num_edges() > 0:
                src, dst = lig_g.edges()
                pos = lig_g.ndata['pos']
                dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
                L_E_bond = dgl.readout_edges(lig_g, model.ligand_bond_sim(dist), op='mean')

            L_E_angle, L_E_torsion = model._calc_angle_energy(lig_g, model.ligand_angle_sim, lig_intra)

            P_E_bond = torch.zeros(1, 1, device=device)
            if prot_g.num_edges() > 0:
                src, dst = prot_g.edges()
                pos = prot_g.ndata['pos']
                dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
                P_E_bond = dgl.readout_edges(prot_g, model.protein_bond_sim(dist), op='mean')

            prot_h = prot_g.ndata["h"]
            prot_intra = model.protein_atom_intra_encoder(prot_g, prot_h)
            P_E_angle, P_E_torsion = model._calc_angle_energy(prot_g, model.protein_angle_sim, prot_intra)

            I_vdw = torch.zeros(1, 1, device=device)
            I_elec = torch.zeros(1, 1, device=device)
            I_hbond = torch.zeros(1, 1, device=device)
            if inter_g is not None and inter_g.num_edges() > 0:
                I_vdw, I_elec, I_hbond = model._calc_inter_energy(inter_g, lig_intra, prot_intra)

            phys = torch.cat([L_E_bond, L_E_angle, L_E_torsion,
                             P_E_bond, P_E_angle, P_E_torsion,
                             I_vdw, I_elec, I_hbond], dim=1)

            features_gnn.append(lig_pooled.cpu().numpy().flatten())
            features_phys.append(phys.cpu().numpy().flatten())
            y_all.append(sample['label'])

    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"  Error on {cid}: {type(e).__name__}: {str(e)[:100]}")

elapsed = time.time() - t0
print(f"Done: {len(features_gnn)}/{len(cids)} in {elapsed:.0f}s ({failed} failed)")

X_gnn = np.stack(features_gnn)
X_phys = np.stack(features_phys)
X = np.column_stack([X_gnn, X_phys])
y = np.array(y_all)
print(f"GNN: {X_gnn.shape}, Physics: {X_phys.shape}, Combined: {X.shape}")
print(f"y: mean={y.mean():.3f}, std={y.std():.3f}")

# ── Save raw features for potential future use ──────────────
with open(f"{CASE_DIR}/pre_gru_features.pkl", "wb") as f:
    pickle.dump({"X": X, "y": y, "cids": cids, "X_gnn": X_gnn, "X_phys": X_phys}, f)

# ── Size-residualization ─────────────────────────────────────
# Ligand size is a confound for affinity prediction too
n_lig_atoms = []
for cid in cids:
    g = data[cid].get("ligand_atom_graph")
    n_lig_atoms.append(g.num_nodes() if g is not None else 0)
n_lig_atoms = np.array(n_lig_atoms)

from sklearn.linear_model import LinearRegression
def residualize(X, confound):
    return X - np.column_stack([
        LinearRegression().fit(confound.reshape(-1, 1), X[:, j]).predict(confound.reshape(-1, 1))
        for j in range(X.shape[1])
    ])

X_res = residualize(X, n_lig_atoms)

# ── Train predictors ────────────────────────────────────────
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(name, model, X_data, y_data, scaler=None):
    if scaler:
        X_data = scaler.fit_transform(X_data)
    y_pred = cross_val_predict(model, X_data, y_data, cv=cv, n_jobs=-1)
    rmse = np.sqrt(mean_squared_error(y_data, y_pred))
    mae = mean_absolute_error(y_data, y_pred)
    pearson = np.corrcoef(y_data, y_pred)[0, 1]
    return {"name": name, "RMSE": rmse, "MAE": mae, "Pearson": pearson}

print(f"\n=== Feature: 73-dim (64 GNN + 9 Physics) ===")
print(f"{'Method':30s} | {'RMSE':>8s} | {'MAE':>8s} | {'Pearson':>8s}")
print("-" * 65)
results = []

for name, model, use_scaler in [
    ("Ridge (raw)", Ridge(alpha=1.0), True),
    ("Ridge (residualized)", Ridge(alpha=1.0), True),
    ("RidgeCV (raw)", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]), True),
    ("RidgeCV (residualized)", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]), True),
    ("RF 300 trees (raw)", RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1), True),
    ("RF 300 trees (residualized)", RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1), True),
    ("GBR (raw)", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42), True),
    ("GBR (residualized)", GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42), True),
    ("SVR (raw)", SVR(kernel='rbf', C=10.0), True),
    ("SVR (residualized)", SVR(kernel='rbf', C=10.0), True),
]:
    X_use = X_res if "residualized" in name else X
    r = evaluate(name, model, X_use.copy(), y, StandardScaler() if use_scaler else None)
    print(f"  {r['name']:28s} | {r['RMSE']:8.4f} | {r['MAE']:8.4f} | {r['Pearson']:8.4f}")
    results.append(r)

# ── Summary ─────────────────────────────────────────────────
best = max(results, key=lambda r: r["Pearson"])
print(f"\nBest: {best['name']} | RMSE={best['RMSE']:.4f}, Pearson={best['Pearson']:.4f}")

target_rmse, target_pearson = 0.59, 0.85
if best["RMSE"] <= target_rmse or best["Pearson"] >= target_pearson:
    print("TARGET MET!")
else:
    print(f"Gap to target: RMSE +{best['RMSE']-target_rmse:.2f}, Pearson -{target_pearson-best['Pearson']:.2f}")

with open(f"{CASE_DIR}/pre_gru_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {CASE_DIR}/pre_gru_results.json")
