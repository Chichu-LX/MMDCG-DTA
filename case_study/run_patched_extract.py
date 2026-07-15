#!/usr/bin/env python3
"""Extract pre-GRU features via hooks on original forward, then train predictors."""

import pickle, sys, os, time, json, numpy as np
import torch, torch.nn as nn, dgl
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

# ── Capture intermediate features via hooks ──────────────────
# We hook the methods that produce the features we want:
# 1. ligand_atom_intra_encoder output (LigandAtomChannel)
# 2. _calc_bond_energy_weights output
# 3. _calc_angle_energy output
# 4. _calc_inter_energy output

hook_data = {}

def make_hook(name):
    def hook(module, input, output):
        hook_data[name] = output
    return hook

# Hook ligand_atom_intra_encoder's output
h1 = model.ligand_atom_intra_encoder.register_forward_hook(make_hook("lig_intra"))

# Hook individual energy modules to capture raw energies
# The physics energies are computed inside _calc_bond_energy_weights, _calc_angle_energy, _calc_inter_energy
# These are called from forward(), not as separate modules
# So we need a different approach — wrap these methods

# Simpler approach: wrap the entire forward to return what we need
original_forward = model.forward

# Register hook on pred_fc to capture fusion rep (already tested and working)
fusion_pos = {}
def fusion_post_hook(module, input, output):
    fusion_pos['val'] = input[0].detach().cpu().squeeze().numpy()
model.pred_fc.register_forward_hook(fusion_post_hook)

# Hook on lig_intra encoder output for 64-dim GNN features
gnn_hook_store = {}
def gnn_hook(module, input, output):
    gnn_hook_store['val'] = output  # (total_atoms, 64)
model.ligand_atom_intra_encoder.register_forward_hook(gnn_hook)

# ── Now wrap forward to also return GNN pooled + physics ─────
def wrapped_forward(sample):
    y = original_forward(sample)
    # GNN: mean-pool lig_intra across atoms (store in ndata first, then readout by name)
    lig_intra = gnn_hook_store.get('val')
    lig_graph = sample["ligand_atom_graph"]
    if lig_intra is not None:
        with lig_graph.local_scope():
            lig_graph.ndata['_gnn_tmp'] = lig_intra
            lig_pooled = dgl.readout_nodes(lig_graph, '_gnn_tmp', op='mean')
    else:
        lig_pooled = torch.zeros(64, device=device)

    # Physics: re-run the energy computations (they're deterministic, no side effects)
    lig_g = sample["ligand_atom_graph"]
    prot_g = sample["protein_atom_graph"]
    inter_g = sample.get("atom_interaction_graph")

    # Bond energies
    if lig_g.num_edges() > 0:
        src, dst = lig_g.edges()
        pos = lig_g.ndata['pos']
        dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
        with lig_g.local_scope():
            lig_g.edata['_e'] = model.ligand_bond_sim(dist)
            L_bond = dgl.readout_edges(lig_g, '_e', op='mean')
    else:
        L_bond = torch.zeros(1, 1, device=device)

    L_angle, L_torsion = model._calc_angle_energy(lig_g, model.ligand_angle_sim, lig_intra)

    if prot_g.num_edges() > 0:
        src, dst = prot_g.edges()
        pos = prot_g.ndata['pos']
        dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
        with prot_g.local_scope():
            prot_g.edata['_e'] = model.protein_bond_sim(dist)
            P_bond = dgl.readout_edges(prot_g, '_e', op='mean')
    else:
        P_bond = torch.zeros(1, 1, device=device)

    prot_h = prot_g.ndata["h"]
    prot_intra = model.protein_atom_intra_encoder(prot_g, prot_h)
    P_angle, P_torsion = model._calc_angle_energy(prot_g, model.protein_angle_sim, prot_intra)

    if inter_g is not None and inter_g.num_edges() > 0:
        I_vdw, I_elec, I_hbond = model._calc_inter_energy(inter_g, lig_intra, prot_intra)
    else:
        I_vdw = torch.zeros(1, 1, device=device)
        I_elec = torch.zeros(1, 1, device=device)
        I_hbond = torch.zeros(1, 1, device=device)

    phys = torch.cat([L_bond, L_angle, L_torsion, P_bond, P_angle, P_torsion, I_vdw, I_elec, I_hbond], dim=1)

    gnn_hook_store.clear()
    return y, lig_pooled, phys

model.forward = wrapped_forward

# ── Extract ──────────────────────────────────────────────────
cids = list(data.keys())
feat_gnn, feat_phys, feat_fusion = [], [], []
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
            y_pred, lig_p, phys = model(sample_dev)
            feat_gnn.append(lig_p.cpu().numpy().flatten())
            feat_phys.append(phys.cpu().numpy().flatten())
            feat_fusion.append(fusion_pos.get('val', np.zeros(285)))
            y_all.append(sample['label'])
    except Exception as e:
        failed += 1
        if failed <= 5:
            print(f"  Error {cid}: {type(e).__name__}: {str(e)[:80]}")

elapsed = time.time() - t0
print(f"Done: {len(feat_gnn)}/{len(cids)} in {elapsed:.0f}s ({failed} failed)")

X_gnn = np.stack(feat_gnn)
X_phys = np.stack(feat_phys)
X_fusion = np.stack(feat_fusion)
y = np.array(y_all)

# Size-residualization
n_lig = np.array([data[c].get("ligand_atom_graph").num_nodes() for c in cids[:len(y_all)]])
from sklearn.linear_model import LinearRegression
def res(X, c):
    return X - np.column_stack([LinearRegression().fit(c.reshape(-1,1), X[:,j]).predict(c.reshape(-1,1)) for j in range(X.shape[1])])

# ── Train all combinations ──────────────────────────────────
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

cv = KFold(n_splits=5, shuffle=True, random_state=42)

def ev(name, model, Xd, yd):
    s = StandardScaler()
    Xs = s.fit_transform(Xd)
    yp = cross_val_predict(model, Xs, yd, cv=cv, n_jobs=-1)
    return {"name": name, "RMSE": float(np.sqrt(mean_squared_error(yd, yp))),
            "MAE": float(mean_absolute_error(yd, yp)),
            "Pearson": float(np.corrcoef(yd, yp)[0,1])}

all_r = []
for fn, Xf in [("GNN(64)", X_gnn), ("Phys(9)", X_phys), ("GNN+Phys(73)", np.column_stack([X_gnn, X_phys])),
                ("GNN+Phys+Fusion(358)", np.column_stack([X_gnn, X_phys, X_fusion]))]:
    for fn2, Xf2 in [("raw", Xf), ("res", res(Xf, n_lig))]:
        key = f"{fn} {fn2}"
        r = ev(f"{key} RidgeCV", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]), Xf2, y)
        print(f"  {r['name']:45s} | RMSE={r['RMSE']:.4f}  MAE={r['MAE']:.4f}  Pearson={r['Pearson']:.4f}")
        all_r.append(r)
        r2 = ev(f"{key} RF", RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1), Xf2, y)
        print(f"  {r2['name']:45s} | RMSE={r2['RMSE']:.4f}  MAE={r2['MAE']:.4f}  Pearson={r2['Pearson']:.4f}")
        all_r.append(r2)

best = max(all_r, key=lambda x: x["Pearson"])
print(f"\nBEST: {best['name']} | RMSE={best['RMSE']:.4f}, Pearson={best['Pearson']:.4f}")
print(f"Target: RMSE<0.59, P>0.85 | Gap: RMSE+{best['RMSE']-0.59:.2f}, P-{0.85-best['Pearson']:.2f}")

with open(f"{CASE_DIR}/final_predictor_results.json", "w") as f:
    json.dump(all_r, f, indent=2)
