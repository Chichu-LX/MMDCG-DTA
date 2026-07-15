#!/usr/bin/env python3
"""Robust training with per-sample progress tracking and skip-on-error."""

import os, sys, pickle, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
    print("Robust MMDCG-DTA Training v4")
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
    target_mean = np.mean(labels)
    target_std = np.std(labels)
    print(f"Target: mean={target_mean:.3f}, std={target_std:.3f}, "
          f"range=[{min(labels):.2f}, {max(labels):.2f}], N={len(labels)}")

    all_ids = list(graph_data.keys())
    dataset = HIVDataset(graph_data, all_ids)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_single)

    config = {
        'embedding_dim': 64,
        'd_atom': 4.0, 'd_res': 8.0, 'd_sub': 8.0,
        'l_intra': 2, 'l_inter': 2, 'l_atom': 2, 'l_sub': 2,
        'inter_negative_slope': 0.2,
        'sub_x_dim': 5, 'raw_atom_dim': 5, 'prot_res_dim': 1,
    }

    model = MMDCGDTAModel_Stage1(config).to(device)

    def init_weights(m):
        if isinstance(m, nn.Linear) and hasattr(m, 'weight') and m.weight.dim() >= 2:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.apply(init_weights)

    # SKIP pretrained weights - train from scratch
    print("Training from scratch (no pretrained weights)")
    # Try loading pretrained
    for pp in []:  # DISABLED ["../Data/stage1_model_final.pth", "../Data/stage1_model_best.pth"]:
        if os.path.exists(pp):
            try:
                state = torch.load(pp, map_location=device, weights_only=True)
                model.load_state_dict(state, strict=False)
                print(f"Loaded pretrained: {pp}")
                break
            except Exception as e:
                print(f"Failed to load {pp}: {e}")

    # Higher LR for faster convergence
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                                       patience=8, verbose=True)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    problem_ids = set()  # Track problematic complex IDs

    for epoch in range(1, 151):
        model.train()
        total_loss = 0.0
        skipped = 0
        all_true, all_pred = [], []
        t0 = time.time()
        last_print = t0

        for i, (cid, sample) in enumerate(loader):
            # Progress tracking
            now = time.time()
            if now - last_print > 30:
                elapsed = now - t0
                rate = (i + 1) / max(1, elapsed / 60)
                print(f"  [{i+1}/{len(dataset)}] {elapsed:.0f}s elapsed, ~{rate:.0f} samples/min", flush=True)
                last_print = now

            try:
                if sample.get('label') is None:
                    skipped += 1
                    continue

                raw_label = sample['label']
                y_true = torch.tensor([(raw_label - target_mean) / target_std],
                                      dtype=torch.float32, device=device)

                sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                             for k, v in sample.items() if k != 'label'}
                y_pred = model(sample_dev)

                if torch.isnan(y_pred).any():
                    skipped += 1
                    problem_ids.add(cid)
                    continue

                loss = criterion(y_pred.view(-1), y_true.view(-1))

                if torch.isnan(loss):
                    skipped += 1
                    problem_ids.add(cid)
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss += loss.item()
                all_true.append(raw_label)
                all_pred.append(y_pred.item() * target_std + target_mean)
            except Exception as e:
                skipped += 1
                problem_ids.add(cid)
                if skipped <= 5:
                    print(f"  Error on {cid}: {type(e).__name__}: {str(e)[:80]}", flush=True)

        avg_loss = total_loss / max(1, len(all_true))
        metrics = evaluate_metrics(np.array(all_true), np.array(all_pred)) if all_true else {
            'RMSE': float('nan'), 'MAE': float('nan'), 'Pearson': 0.0, 'SD': float('nan')}

        elapsed = time.time() - t0
        scheduler.step(avg_loss)

        msg = (f"Epoch {epoch:03d} | Loss: {avg_loss:.4f} | "
               f"RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | "
               f"Pearson: {metrics['Pearson']:.4f} | "
               f"Skip: {skipped} | Time: {elapsed:.0f}s")
        print(msg, flush=True)

        if problem_ids:
            print(f"  Problem complexes this epoch: {sorted(problem_ids)}", flush=True)
            problem_ids = set()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_mean': target_mean,
                'target_std': target_std,
                'metrics': metrics,
            }, f"{CASE_DIR}/hiv_protease_best_model.pth")
            print(f"  -> Saved best model", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"Best epoch: {best_epoch}, Best Loss: {best_loss:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
