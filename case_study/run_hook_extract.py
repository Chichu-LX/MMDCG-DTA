#!/usr/bin/env python3
"""Extract fusion features (285-dim) from frozen encoder via hooks, then train predictor."""

import pickle, sys, os, time, json, numpy as np
import torch, dgl
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))
from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"
device = torch.device("cuda")

# Load
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
print(f"Model loaded ({len(state)} keys)")

# ── Hook to capture fusion representation ───────────────────
fusion_store = {}
def fusion_hook(module, input, output):
    # input[0] is the fusion_rep (1, 285) before pred_fc
    fusion_store['data'] = input[0].detach().cpu().squeeze().numpy()

handle = model.pred_fc.register_forward_hook(fusion_hook)

# ── Inference + feature capture ─────────────────────────────
cids = list(data.keys())
y_true_all = []
features_all = []
failed = 0
t0 = time.time()

for i, cid in enumerate(cids):
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(cids)}] {time.time()-t0:.0f}s")
    sample = data[cid]
    try:
        fusion_store.clear()
        with torch.no_grad():
            sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                         for k, v in sample.items() if k != 'label'}
            _ = model(sample_dev)
            feat = fusion_store.get('data')
            if feat is None:
                failed += 1
                continue
            features_all.append(feat)
            y_true_all.append(sample['label'])
    except Exception as e:
        failed += 1

handle.remove()
elapsed = time.time() - t0
print(f"Done: {len(features_all)}/{len(cids)} in {elapsed:.0f}s ({failed} failed)")

X = np.stack(features_all)  # (N, 285)
y = np.array(y_true_all)
print(f"Features: {X.shape}, Labels: {y.shape}")
print(f"y: mean={y.mean():.3f}, std={y.std():.3f}, range=[{y.min():.2f}, {y.max():.2f}]")

# ── Train predictors with 5-fold CV ─────────────────────────
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

scaler = StandardScaler()
X_s = scaler.fit_transform(X)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def evaluate(name, model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    pearson = np.corrcoef(y, y_pred)[0, 1]
    print(f"  {name:25s} | RMSE={rmse:.4f}  MAE={mae:.4f}  Pearson={pearson:.4f}")
    return {"name": name, "RMSE": rmse, "MAE": mae, "Pearson": pearson}

print(f"\n=== Predictor Performance (5-fold CV, N={len(y)}) ===")
print("-" * 70)
results = []

results.append(evaluate("Ridge alpha=1.0", Ridge(alpha=1.0), X_s, y))
results.append(evaluate("RidgeCV", RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0]), X_s, y))
results.append(evaluate("RF 300 trees", RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1), X_s, y))
results.append(evaluate("GBR 200 trees", GradientBoostingRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42), X_s, y))
results.append(evaluate("MLP 128-64", MLPRegressor(
    hidden_layer_sizes=(128, 64), activation='relu', alpha=0.001,
    max_iter=2000, early_stopping=True, random_state=42), X_s, y))

# ── Summary ─────────────────────────────────────────────────
best = max(results, key=lambda r: r["Pearson"])
print(f"\nBest: {best['name']} | RMSE={best['RMSE']:.4f}, Pearson={best['Pearson']:.4f}")

target_rmse, target_pearson = 0.59, 0.85
print(f"Target: RMSE < {target_rmse}, Pearson > {target_pearson}")
if best["RMSE"] <= target_rmse or best["Pearson"] >= target_pearson:
    print("APPROACHING TARGET!")
else:
    print(f"Gap: RMSE off by {best['RMSE']-target_rmse:.2f}, Pearson off by {target_pearson-best['Pearson']:.2f}")

with open(f"{CASE_DIR}/hook_predictor_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {CASE_DIR}/hook_predictor_results.json")
