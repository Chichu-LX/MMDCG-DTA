#!/usr/bin/env python3
"""Quick evaluation of pretrained model on all 330 complexes (no training)."""

import os, sys, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import dgl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))
from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

CASE_DIR = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study"


def evaluate_metrics(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    vx = y_true - np.mean(y_true)
    vy = y_pred - np.mean(y_pred)
    pearson = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
    sd = np.std(y_true - y_pred)
    return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "SD": sd}


class HIVDataset(Dataset):
    def __init__(self, graph_data, ids):
        self.samples = [(cid, graph_data[cid]) for cid in ids if graph_data[cid].get('label') is not None]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_single(batch):
    return batch[0]


def main():
    print("=" * 60)
    print("Quick Evaluation of Pretrained Model on 330 Complexes")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for graph_path in ["hiv_protease_graphs_full.pkl", "hiv_protease_graphs.pkl"]:
        if os.path.exists(graph_path):
            with open(graph_path, 'rb') as f:
                graph_data = pickle.load(f)
            print(f"Loaded {len(graph_data)} graphs from {graph_path}")
            break

    labels = [g['label'] for g in graph_data.values() if g.get('label') is not None]
    print(f"Target: mean={np.mean(labels):.3f}, std={np.std(labels):.3f}, "
          f"range=[{min(labels):.2f}, {max(labels):.2f}], N={len(labels)}")

    config = {
        'embedding_dim': 64,
        'd_atom': 4.0, 'd_res': 8.0, 'd_sub': 8.0,
        'l_intra': 2, 'l_inter': 2, 'l_atom': 2, 'l_sub': 2,
        'inter_negative_slope': 0.2,
        'sub_x_dim': 5, 'raw_atom_dim': 5, 'prot_res_dim': 1,
    }

    model = MMDCGDTAModel_Stage1(config).to(device)
    model.eval()

    # Load pretrained
    loaded = False
    for pp in ["../Data/stage1_model_final.pth", "../Data/stage1_model_best.pth"]:
        if os.path.exists(pp):
            state = torch.load(pp, map_location=device, weights_only=True)
            model.load_state_dict(state, strict=False)
            print(f"Loaded pretrained: {pp}")
            loaded = True
            break

    if not loaded:
        print("ERROR: No pretrained model found!")
        return

    all_ids = list(graph_data.keys())
    dataset = HIVDataset(graph_data, all_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_single)

    all_true, all_pred = [], []
    skipped = 0

    with torch.no_grad():
        for i, (cid, sample) in enumerate(loader):
            try:
                if sample.get('label') is None:
                    skipped += 1
                    continue
                raw_label = sample['label']
                sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                             for k, v in sample.items() if k != 'label'}
                y_pred = model(sample_dev)

                if torch.isnan(y_pred).any():
                    skipped += 1
                    continue

                all_true.append(raw_label)
                all_pred.append(y_pred.item())
            except Exception as e:
                skipped += 1

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(dataset)}...", flush=True)

    metrics = evaluate_metrics(np.array(all_true), np.array(all_pred))
    print(f"\n{'='*60}")
    print(f"Pretrained Model Results (N={len(all_true)}, skipped={skipped}):")
    print(f"  RMSE:   {metrics['RMSE']:.4f}")
    print(f"  MAE:    {metrics['MAE']:.4f}")
    print(f"  Pearson: {metrics['Pearson']:.4f}")
    print(f"  SD:     {metrics['SD']:.4f}")
    print(f"{'='*60}")

    pred_dict = {}
    for i, (cid, _) in enumerate(dataset.samples):
        if i < len(all_true):
            pred_dict[cid] = {'true': all_true[i], 'pred': all_pred[i]}

    with open(f"{CASE_DIR}/pretrained_predictions.pkl", "wb") as f:
        pickle.dump(pred_dict, f)
    print(f"Saved predictions to pretrained_predictions.pkl")


if __name__ == "__main__":
    main()
