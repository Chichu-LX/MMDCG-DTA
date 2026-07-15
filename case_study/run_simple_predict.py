#!/usr/bin/env python3
"""Simple: run frozen model inference on all 330 complexes, evaluate metrics."""

import pickle, sys, os, time, numpy as np
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

# Inference
cids = list(data.keys())
y_true = []
y_pred = []
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
            pred = model(sample_dev)
            y_true.append(sample['label'])
            y_pred.append(pred.item())
    except Exception as e:
        failed += 1

elapsed = time.time() - t0
print(f"Done: {len(y_pred)}/{len(cids)} in {elapsed:.0f}s ({failed} failed)")

# Metrics
y_true = np.array(y_true)
y_pred = np.array(y_pred)
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
pearson = np.corrcoef(y_true, y_pred)[0, 1]

print(f"\n=== Raw Pretrained Model (no fine-tuning) ===")
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, Pearson: {pearson:.4f}")
print(f"y_true: mean={y_true.mean():.2f}, std={y_true.std():.2f}")
print(f"y_pred: mean={y_pred.mean():.2f}, std={y_pred.std():.2f}")

# Save
results = {"RMSE": float(rmse), "MAE": float(mae), "Pearson": float(pearson),
           "y_true": y_true.tolist(), "y_pred": y_pred.tolist(), "cids": cids[:len(y_true)]}
import json
with open(f"{CASE_DIR}/inference_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {CASE_DIR}/inference_results.json")
