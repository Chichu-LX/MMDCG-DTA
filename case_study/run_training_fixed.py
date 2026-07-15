#!/usr/bin/env python3
"""Fixed MMDCG-DTA training — filters broken graphs, proper pretrained loading, validation."""

import os, sys, pickle, json, time, gc
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
os.makedirs(CASE_DIR, exist_ok=True)


def evaluate_metrics(y_true, y_pred):
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mae = float(np.mean(np.abs(yt - yp)))
    vx = yt - yt.mean()
    vy = yp - yp.mean()
    denom = np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2))
    pearson = float(np.sum(vx * vy) / (denom + 1e-8))
    return {"RMSE": rmse, "MAE": mae, "Pearson": pearson}


class HIVDataset(Dataset):
    def __init__(self, graph_data, ids):
        self.samples = [(cid, graph_data[cid]) for cid in ids
                       if graph_data[cid].get('label') is not None]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_single(batch):
    return batch[0]


SKIP_KEYS = {"gru.", "pred_fc."}  # Re-init prediction head, load encoders only

def load_pretrained(model, path, device):
    """Load pretrained ENCODER weights; skip prediction head (gru, pred_fc)."""
    state = torch.load(path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    loaded, skipped, mismatched = [], [], []
    for k, v in state.items():
        if any(k.startswith(s) for s in SKIP_KEYS):
            skipped.append(k)
            continue
        if k in model_state:
            if model_state[k].shape == v.shape:
                model_state[k].copy_(v)
                loaded.append(k)
            else:
                mismatched.append(f"{k}: {v.shape} vs {model_state[k].shape}")
    model.load_state_dict(model_state)
    print(f"Loaded {len(loaded)} encoder keys (skipped {len(skipped)} head keys)")
    if mismatched:
        print(f"  Shape mismatch: {mismatched[:5]}")
    return loaded


def main():
    import logging
    log_path = f"{CASE_DIR}/training_fixed.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s %(message)s")
    logger = logging.getLogger()
    # Also print to stdout
    import sys
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    print("=" * 60)
    print("Fixed MMDCG-DTA Training for HIV-1 Protease")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load graphs ──────────────────────────────────────────────
    for gp in ["hiv_protease_graphs_full.pkl", "hiv_protease_graphs.pkl"]:
        if os.path.exists(gp):
            with open(gp, 'rb') as f:
                graph_data = pickle.load(f)
            print(f"Loaded {len(graph_data)} graphs from {gp}")
            break

    # Use ALL complexes (zero-edge sub-interaction graphs still produce valid outputs)
    all_ids = list(graph_data.keys())

    # ── Train/val split (stratified by label bins) ───────────────
    labels_all = np.array([graph_data[c]["label"] for c in all_ids])
    target_mean = float(np.mean(labels_all))
    target_std = float(np.std(labels_all))
    print(f"Target: mean={target_mean:.3f}, std={target_std:.3f}, "
          f"range=[{labels_all.min():.2f}, {labels_all.max():.2f}], N={len(all_ids)}")

    # Stratified split by label quartiles
    bins = np.quantile(labels_all, [0, 0.25, 0.5, 0.75, 1.0])
    bin_idx = np.digitize(labels_all, bins[:-1]) - 1
    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(
        all_ids, test_size=0.15, random_state=42, stratify=bin_idx)
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    train_dataset = HIVDataset(graph_data, train_ids)
    val_dataset = HIVDataset(graph_data, val_ids)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_single)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_single)

    # ── Model ────────────────────────────────────────────────────
    config = {
        'embedding_dim': 64, 'd_atom': 4.0, 'd_res': 8.0, 'd_sub': 8.0,
        'l_intra': 2, 'l_inter': 2, 'l_atom': 2, 'l_sub': 2,
        'inter_negative_slope': 0.2,
        'sub_x_dim': 5, 'raw_atom_dim': 5, 'prot_res_dim': 1,
    }
    model = MMDCGDTAModel_Stage1(config).to(device)

    # Load pretrained WITHOUT re-initializing first
    pretrained_loaded = False
    for pp in ["../Data/stage1_model_final.pth", "../Data/stage1_model_best.pth"]:
        if os.path.exists(pp):
            load_pretrained(model, pp, device)
            pretrained_loaded = True
            break

    if not pretrained_loaded:
        print("WARNING: No pretrained model found — training from scratch")

    # ── Optimizer: higher LR for new prediction head, lower for pretrained encoders ──
    head_params = []
    encoder_params = []
    for n, p in model.named_parameters():
        if any(n.startswith(s) for s in ["gru.", "pred_fc.", "frag_proj.", "res_proj."]):
            head_params.append(p)
        else:
            encoder_params.append(p)
    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": 1e-5},
        {"params": head_params, "lr": 5e-4},
    ], weight_decay=1e-5)
    print(f"Encoder params: {len(encoder_params)}, Head params: {len(head_params)}")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    criterion = nn.MSELoss()

    # ── Training loop ────────────────────────────────────────────
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    MAX_PATIENCE = 50

    for epoch in range(1, 201):
        model.train()
        total_loss = 0.0
        all_true_tr, all_pred_tr = [], []
        t0 = time.time()

        for cid, sample in train_loader:
            try:
                raw_label = sample['label']
                y_true = torch.tensor([(raw_label - target_mean) / target_std],
                                      dtype=torch.float32, device=device)

                sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                             for k, v in sample.items() if k != 'label'}
                y_pred = model(sample_dev)

                if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                    continue

                loss = criterion(y_pred.view(-1), y_true.view(-1))
                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()

                total_loss += loss.item()
                all_true_tr.append(raw_label)
                all_pred_tr.append(y_pred.item() * target_std + target_mean)
            except Exception as e:
                continue

        # ── Validation ───────────────────────────────────────────
        model.eval()
        val_loss_total = 0.0
        all_true_val, all_pred_val = [], []
        with torch.no_grad():
            for cid, sample in val_loader:
                try:
                    raw_label = sample['label']
                    y_true = torch.tensor([(raw_label - target_mean) / target_std],
                                          dtype=torch.float32, device=device)

                    sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                                 for k, v in sample.items() if k != 'label'}
                    y_pred = model(sample_dev)

                    if torch.isnan(y_pred).any():
                        continue

                    loss = criterion(y_pred.view(-1), y_true.view(-1))
                    val_loss_total += loss.item()
                    all_true_val.append(raw_label)
                    all_pred_val.append(y_pred.item() * target_std + target_mean)
                except Exception as e:
                    continue

        # ── Metrics ───────────────────────────────────────────────
        n_train = len(all_true_tr)
        n_val = len(all_true_val)
        avg_train_loss = total_loss / max(1, n_train)
        avg_val_loss = val_loss_total / max(1, n_val)

        tr_metrics = evaluate_metrics(all_true_tr, all_pred_tr) if n_train > 0 else {"RMSE": float('nan'), "MAE": float('nan'), "Pearson": 0.0}
        val_metrics = evaluate_metrics(all_true_val, all_pred_val) if n_val > 0 else {"RMSE": float('nan'), "MAE": float('nan'), "Pearson": 0.0}

        elapsed = time.time() - t0
        scheduler.step(avg_val_loss)

        msg = (f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | "
               f"Val Loss: {avg_val_loss:.4f} | "
               f"Train RMSE: {tr_metrics['RMSE']:.4f} P: {tr_metrics['Pearson']:.3f} | "
               f"Val RMSE: {val_metrics['RMSE']:.4f} P: {val_metrics['Pearson']:.3f} | "
               f"Time: {elapsed:.0f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
        print(msg)
        logger.info(msg)

        # ── Save best ────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'target_mean': target_mean, 'target_std': target_std,
                'train_metrics': tr_metrics, 'val_metrics': val_metrics,
            }, f"{CASE_DIR}/hiv_protease_best_model_fixed.pth")
            print(f"  -> Saved best model (Val RMSE={val_metrics['RMSE']:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= MAX_PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\n{'='*60}")
    print(f"Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    print(f"Final Val RMSE: {val_metrics['RMSE']:.4f}, Val Pearson: {val_metrics['Pearson']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
