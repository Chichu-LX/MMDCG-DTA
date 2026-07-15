"""
Master Pipeline: HIV-1 Protease Case Study with MMDCG-DTA.

Complete workflow:
  Step 1: Extract HIV-1 Protease data from PDBbind
  Step 2: Build MMDCG-DTA-compatible graph dataset
  Step 3: Fine-tune MMDCG-DTA model on HIV PR data
  Step 4: Run inference with full interpretability analysis

Uses MMDCGDTAModel_Stage1 from Data/MMDCG_DTA_Stage1.py which matches the
pretrained checkpoint architecture (stage1_model_best.pth).
"""

import os
import sys
import yaml
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import time
import math
from torch.utils.data import Dataset, DataLoader
import dgl

# Add parent directory and Data directory for module imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Data'))

from Data.MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1

# Import metrics
try:
    from Utils.metrics import evaluate_metrics
except ImportError:
    def evaluate_metrics(y_true, y_pred):
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae = np.mean(np.abs(y_true - y_pred))
        vx = y_true - np.mean(y_true)
        vy = y_pred - np.mean(y_pred)
        std_x = np.std(y_true)
        std_y = np.std(y_pred)
        if std_x < 1e-6 or std_y < 1e-6:
            pearson = 0.0
        else:
            pearson = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)) + 1e-8)
        error = y_true - y_pred
        sd = np.std(error)
        return {"RMSE": rmse, "MAE": mae, "Pearson": pearson, "SD": sd}

# Import local modules
from build_hiv_graphs import (
    build_ligand_atom_graph, build_protein_atom_graph,
    build_atom_interaction_graph, build_ligand_fragment_graph,
    build_protein_residue_graph, build_substructure_interaction_graph,
    build_all_graphs, safe_read_sdf, patch_group_ids
)
from inference_interpretability import (
    MMDCGDTAInterpretable, analyze_graph_topology,
    compute_interpretability_metrics, run_inference
)


# ============================================================================
# Dataset for Training
# ============================================================================

class HIVProteaseDataset(Dataset):
    def __init__(self, graph_data, ids):
        self.samples = []
        for cid in ids:
            sample = graph_data[cid]
            if sample.get('label') is not None:
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_single(batch):
    return batch[0]


# ============================================================================
# Training
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_true, all_pred = [], []
    skipped = 0

    for sample in loader:
        try:
            if sample.get('label') is None:
                skipped += 1
                continue

            y_true = torch.tensor([sample['label']], dtype=torch.float32, device=device)

            # Move graphs to device
            sample_dev = {}
            for k, v in sample.items():
                if hasattr(v, 'to'):
                    sample_dev[k] = v.to(device)
                else:
                    sample_dev[k] = v

            y_pred = model(sample_dev)

            if torch.isnan(y_pred).any():
                skipped += 1
                continue

            loss = criterion(y_pred.view(-1), y_true.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            all_true.append(y_true.item())
            all_pred.append(y_pred.item())
        except Exception as e:
            skipped += 1

    avg_loss = total_loss / max(1, len(all_true))
    if all_true:
        metrics = evaluate_metrics(np.array(all_true), np.array(all_pred))
    else:
        metrics = {'RMSE': float('nan'), 'MAE': float('nan'),
                   'Pearson': 0.0, 'SD': float('nan')}
    return avg_loss, metrics


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_true, all_pred = [], []

    for sample in loader:
        try:
            if sample.get('label') is None:
                continue
            y_true = torch.tensor([sample['label']], dtype=torch.float32, device=device)
            sample_dev = {k: v.to(device) if hasattr(v, 'to') else v
                         for k, v in sample.items()}
            y_pred = model(sample_dev)
            if torch.isnan(y_pred).any():
                continue
            loss = criterion(y_pred.view(-1), y_true.view(-1))
            total_loss += loss.item()
            all_true.append(y_true.item())
            all_pred.append(y_pred.item())
        except:
            pass

    avg_loss = total_loss / max(1, len(all_true))
    if all_true:
        metrics = evaluate_metrics(np.array(all_true), np.array(all_pred))
    else:
        metrics = {'RMSE': float('nan'), 'MAE': float('nan'),
                   'Pearson': 0.0, 'SD': float('nan')}
    return avg_loss, metrics


def fine_tune_model(graph_data, config, device):
    """
    Fine-tune MMDCG-DTA on HIV-1 Protease data.
    Uses leave-one-out cross-validation given the small dataset (~50 complexes).
    """
    complex_ids = list(graph_data.keys())
    print(f"\nFine-tuning on {len(complex_ids)} HIV-1 Protease complexes")
    print(f"{'='*60}")

    # Use ALL complexes for training (no held-out validation)
    train_ids = complex_ids
    val_ids = complex_ids

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)} (all data used for training)")

    train_dataset = HIVProteaseDataset(graph_data, train_ids)
    val_dataset = HIVProteaseDataset(graph_data, val_ids)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                               collate_fn=collate_single)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                             collate_fn=collate_single)

    # Initialize model (MMDCGDTAModel_Stage1 - matches checkpoint architecture)
    model = MMDCGDTAModel_Stage1(config).to(device)

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear) and hasattr(m, 'weight') and m.weight.dim() >= 2:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_weights)

    # Load pretrained Stage 1 weights
    pretrained_paths = [
        "../Data/stage1_model_final.pth",
        "../Data/stage1_model_best.pth",
    ]

    loaded = False
    for pp in pretrained_paths:
        if os.path.exists(pp):
            try:
                state = torch.load(pp, map_location=device, weights_only=True)
                model.load_state_dict(state, strict=False)
                print(f"Loaded pretrained weights from {pp}")
                loaded = True
                break
            except Exception as e:
                print(f"Could not load {pp}: {e}")

    if not loaded:
        print("WARNING: No pretrained weights found. Using random initialization.")

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'],
                           weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = nn.MSELoss()

    best_val_rmse = float('inf')
    patience_counter = 0
    log_lines = []

    for epoch in range(1, config['max_epochs'] + 1):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        msg = (f"Epoch {epoch:03d} | "
               f"Train Loss: {train_loss:.4f} | "
               f"Val RMSE: {val_metrics['RMSE']:.4f} | "
               f"Val Pearson: {val_metrics['Pearson']:.3f}")
        print(msg)
        log_lines.append(msg)

        if val_metrics['RMSE'] < best_val_rmse:
            best_val_rmse = val_metrics['RMSE']
            patience_counter = 0
            torch.save(model.state_dict(), "hiv_protease_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print("Early stopping")
                break

    with open("hiv_protease_training_log.txt", 'w') as f:
        f.write('\n'.join(log_lines))

    print(f"\nBest Val RMSE: {best_val_rmse:.4f}")

    return model


# ============================================================================
# Full Pipeline
# ============================================================================

def main():
    print("=" * 60)
    print("MMDCG-DTA Case Study: HIV-1 Protease Binding Affinity Prediction")
    print("=" * 60)

    # Load config
    with open("case_study_config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: MMDCGDTAModel_Stage1 (embedding_dim={config['embedding_dim']})")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ===================================================================
    # Step 1: Extract raw data
    # ===================================================================
    raw_data_path = "hiv_protease_raw.pkl"
    if not os.path.exists(raw_data_path):
        print("\n[Step 1] Extracting HIV-1 Protease data from PDBbind...")
        from extract_hiv_protease_data import extract_hiv_protease_data
        pdbbind_base = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/Data/PDBbind_dataset"
        extract_hiv_protease_data(pdbbind_base, ".")
    else:
        with open(raw_data_path, 'rb') as f:
            raw_data = pickle.load(f)
        print(f"\n[Step 1] Loaded {len(raw_data)} raw complexes from {raw_data_path}")

    # ===================================================================
    # Step 2: Build graphs (with K-Means group assignment)
    # ===================================================================
    graph_path = "hiv_protease_graphs.pkl"
    if not os.path.exists(graph_path):
        print("\n[Step 2] Building graph dataset...")
        with open(raw_data_path, 'rb') as f:
            raw_data = pickle.load(f)
        graph_data = build_all_graphs(raw_data, config, verbose=True)
        with open(graph_path, 'wb') as f:
            pickle.dump(graph_data, f)
        print(f"Saved {len(graph_data)} graph complexes to {graph_path}")
    else:
        with open(graph_path, 'rb') as f:
            graph_data = pickle.load(f)
        print(f"\n[Step 2] Loaded {len(graph_data)} graph complexes from {graph_path}")

    # ===================================================================
    # Step 3: Fine-tune model
    # ===================================================================
    print("\n[Step 3] Fine-tuning MMDCG-DTA model on HIV-1 Protease data...")
    model = fine_tune_model(graph_data, config, device)

    # ===================================================================
    # Step 4: Inference with interpretability
    # ===================================================================
    print(f"\n[Step 4] Running full inference with interpretability analysis...")
    print("=" * 60)

    # Load best model into interpretable wrapper
    best_model_path = "hiv_protease_best_model.pth"
    model_interp = MMDCGDTAInterpretable(config).to(device)
    if os.path.exists(best_model_path):
        model_interp.load_state_dict(
            torch.load(best_model_path, map_location=device, weights_only=True),
            strict=False
        )
        print("Loaded best fine-tuned model for inference")
    else:
        print("WARNING: No fine-tuned model found. Using pretrained weights.")
        pretrained_path = "../Data/stage1_model_best.pth"
        if os.path.exists(pretrained_path):
            state = torch.load(pretrained_path, map_location=device, weights_only=True)
            model_interp.load_state_dict(state, strict=False)
            print("Loaded pretrained Stage 1 weights")

    results, all_predictions = run_inference(model_interp, graph_data, device, config)

    # ===================================================================
    # Step 5: Generate report
    # ===================================================================
    print(f"\n[Step 5] Generating interpretability report...")
    print("=" * 60)

    true_vals = np.array([p[0] for p in all_predictions])
    pred_vals = np.array([p[1] for p in all_predictions])
    metrics = evaluate_metrics(true_vals, pred_vals)

    # Save comprehensive results
    import json
    output = {
        'case_study': 'HIV-1 Protease',
        'num_complexes': len(graph_data),
        'overall_metrics': {k: float(v) for k, v in metrics.items()},
        'per_complex': {},
        'graph_topology_summary': {},
        'interpretability_analysis': {},
    }

    # Per-complex details
    sorted_cases = sorted(results.items(), key=lambda x: x[1]['true_pKd'])
    for comp_id, res in sorted_cases:
        output['per_complex'][comp_id] = {
            'true_pKd': float(res['true_pKd']),
            'predicted_pKd': float(res['predicted_pKd']),
            'error': float(res['error']),
            'absolute_error': float(abs(res['error'])),
            'graph_topology': res['graph_topology'],
            'interpretability': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in res['interpretability'].items()
            },
        }

    # Graph topology aggregation
    all_lig_atoms = []
    all_prot_atoms = []
    all_frags = []
    all_residues = []
    all_int_edges = []

    for comp_id, res in results.items():
        topo = res['graph_topology']
        all_lig_atoms.append(topo['ligand_atom_graph']['num_nodes'])
        all_prot_atoms.append(topo['protein_atom_graph']['num_nodes'])
        all_frags.append(topo['ligand_fragment_graph']['num_nodes'])
        all_residues.append(topo['protein_residue_graph']['num_nodes'])
        if 'atom_interaction_graph' in topo:
            all_int_edges.append(topo['atom_interaction_graph']['num_edges'])

    output['graph_topology_summary'] = {
        'ligand_atoms': {'mean': float(np.mean(all_lig_atoms)), 'std': float(np.std(all_lig_atoms)),
                         'min': int(np.min(all_lig_atoms)), 'max': int(np.max(all_lig_atoms))},
        'protein_atoms': {'mean': float(np.mean(all_prot_atoms)), 'std': float(np.std(all_prot_atoms)),
                          'min': int(np.min(all_prot_atoms)), 'max': int(np.max(all_prot_atoms))},
        'ligand_fragments': {'mean': float(np.mean(all_frags)), 'std': float(np.std(all_frags)),
                             'min': int(np.min(all_frags)), 'max': int(np.max(all_frags))},
        'protein_residues': {'mean': float(np.mean(all_residues)), 'std': float(np.std(all_residues)),
                             'min': int(np.min(all_residues)), 'max': int(np.max(all_residues))},
        'interaction_edges': {'mean': float(np.mean(all_int_edges)) if all_int_edges else 0,
                              'std': float(np.std(all_int_edges)) if all_int_edges else 0,
                              'min': int(np.min(all_int_edges)) if all_int_edges else 0,
                              'max': int(np.max(all_int_edges)) if all_int_edges else 0},
    }

    # Affinity-stratified analysis
    low_binders = [c for c in sorted_cases if c[1]['true_pKd'] < 7]
    med_binders = [c for c in sorted_cases if 7 <= c[1]['true_pKd'] < 9]
    high_binders = [c for c in sorted_cases if c[1]['true_pKd'] >= 9]

    def _compute_prediction_stats(cases):
        if not cases:
            return {}
        errors = [c[1]['error'] for c in cases]
        abs_errors = [abs(e) for e in errors]
        return {
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'mae': float(np.mean(abs_errors)),
            'rmse': float(np.sqrt(np.mean(np.array(errors)**2))),
        }

    output['affinity_stratification'] = {
        'low_binders (pKd<7)': {
            'count': len(low_binders),
            'prediction_stats': _compute_prediction_stats(low_binders),
        },
        'medium_binders (7<=pKd<9)': {
            'count': len(med_binders),
            'prediction_stats': _compute_prediction_stats(med_binders),
        },
        'high_binders (pKd>=9)': {
            'count': len(high_binders),
            'prediction_stats': _compute_prediction_stats(high_binders),
        },
    }

    # Save reports
    report_path = "hiv_protease_case_study_report.json"
    with open(report_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")

    repr_path = "hiv_protease_representations.pkl"
    with open(repr_path, 'wb') as f:
        pickle.dump(results, f)
    print(f"Full representations saved to {repr_path}")

    # ===================================================================
    # Final Summary
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"CASE STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"\nTarget: HIV-1 Protease")
    print(f"Complexes analyzed: {len(graph_data)}")
    print(f"\nPrediction Performance:")
    print(f"  RMSE:      {metrics['RMSE']:.4f} pKd")
    print(f"  MAE:       {metrics['MAE']:.4f} pKd")
    print(f"  Pearson R: {metrics['Pearson']:.4f}")
    print(f"  SD:        {metrics['SD']:.4f} pKd")
    print(f"\nOutput files:")
    print(f"  - hiv_protease_raw.pkl                 (raw complex data)")
    print(f"  - hiv_protease_graphs.pkl              (DGL graph data)")
    print(f"  - hiv_protease_best_model.pth          (fine-tuned model)")
    print(f"  - hiv_protease_case_study_report.json   (comprehensive report)")
    print(f"  - hiv_protease_representations.pkl     (full representations)")
    print(f"  - hiv_protease_training_log.txt        (training log)")


if __name__ == "__main__":
    main()
