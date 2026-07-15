"""
Multi-Stage MMDCG-DTA Case Study Pipeline.
Runs inference with Stage 1, Stage 2, Stage 3 models on HIV-1 Protease data,
compares results, and generates comprehensive multi-stage analysis.

Usage (on server):
  export LD_LIBRARY_PATH=/root/anaconda3/envs/mmdcg_dta_env/lib:$LD_LIBRARY_PATH
  conda activate /root/anaconda3/envs/mmdcg_dta_env
  cd /root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main/case_study
  python run_multistage_pipeline.py
"""

import os
import sys
import yaml
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
import random
from collections import defaultdict

# Server paths
SERVER_BASE = "/root/protein_ligand/MMDCG-DTA/MMDCG-DTA-main"
DATA_PATH = os.path.join(SERVER_BASE, "Data")
sys.path.insert(0, DATA_PATH)
sys.path.insert(0, os.path.join(SERVER_BASE, "case_study"))

from inference_multistage import (
    MMDCGDTAInterpretable_Stage1, MMDCGDTAInterpretable_Stage2, MMDCGDTAInterpretable_Stage3,
    run_multistage_inference, evaluate_metrics
)


def load_config():
    config_path = os.path.join(SERVER_BASE, "case_study", "case_study_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_graph_data():
    graph_path = os.path.join(SERVER_BASE, "case_study", "hiv_protease_graphs.pkl")
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def build_model(config, stage, device):
    if stage == 1:
        model = MMDCGDTAInterpretable_Stage1(config)
    elif stage == 2:
        model = MMDCGDTAInterpretable_Stage2(config)
    elif stage == 3:
        model = MMDCGDTAInterpretable_Stage3(config)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return model.to(device)


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint with strict=False and report missing/unexpected keys."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
    return model


def main():
    print("=" * 60)
    print("MMDCG-DTA Multi-Stage Case Study: HIV-1 Protease")
    print("=" * 60)

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: embedding_dim={config['embedding_dim']}, "
          f"d_atom={config['d_atom']}, d_sub={config['d_sub']}")

    # Load graph data
    print("\nLoading graph data...")
    graph_data = load_graph_data()
    print(f"Loaded {len(graph_data)} complexes")

    # =====================================================================
    # Stage 1: Load pretrained + fine-tuned checkpoint
    # =====================================================================
    print("\n" + "=" * 40)
    print("STAGE 1: Physics-Informed Base Model")
    print("=" * 40)

    model_s1 = build_model(config, 1, device)
    s1_checkpoint = os.path.join(SERVER_BASE, "case_study", "hiv_protease_best_model.pth")
    if os.path.exists(s1_checkpoint):
        model_s1 = load_checkpoint(model_s1, s1_checkpoint, device)
        print(f"Loaded Stage 1 fine-tuned checkpoint: {s1_checkpoint}")
    else:
        s1_ckpt = os.path.join(DATA_PATH, "stage1_model_best.pth")
        if os.path.exists(s1_ckpt):
            model_s1 = load_checkpoint(model_s1, s1_ckpt, device)
            print(f"Loaded Stage 1 pretrained checkpoint: {s1_ckpt}")
        else:
            print("WARNING: No Stage 1 checkpoint found!")

    # =====================================================================
    # Stage 2: Load edge reconstruction model
    # =====================================================================
    print("\n" + "=" * 40)
    print("STAGE 2: Edge Reconstruction Model")
    print("=" * 40)

    model_s2 = build_model(config, 2, device)
    s2_checkpoints = [
        os.path.join(DATA_PATH, "Model", "Stage2", "stage2_model_final.pth"),
        os.path.join(DATA_PATH, "Model", "Stage3", "stage2_model_epoch_50.pth"),
        os.path.join(DATA_PATH, "Model", "Stage3", "stage2_model_epoch_40.pth"),
        os.path.join(DATA_PATH, "first_finished_code", "stage2_model_final.pth"),
    ]
    loaded_s2 = False
    for ckpt in s2_checkpoints:
        if os.path.exists(ckpt):
            model_s2 = load_checkpoint(model_s2, ckpt, device)
            print(f"Loaded Stage 2 checkpoint: {ckpt}")
            loaded_s2 = True
            break
    if not loaded_s2:
        print("WARNING: No Stage 2 checkpoint found, using Stage 1 weights")
        s1_state = model_s1.state_dict()
        model_s2.load_state_dict(s1_state, strict=False)

    # =====================================================================
    # Stage 3: Fine-tuned with frozen edge reconstructor
    # =====================================================================
    print("\n" + "=" * 40)
    print("STAGE 3: Fine-tuned with Frozen Reconstructor")
    print("=" * 40)

    model_s3 = build_model(config, 3, device)
    s3_checkpoints = [
        os.path.join(DATA_PATH, "Model", "Stage3", "stage3_model_best.pth"),
        os.path.join(DATA_PATH, "Model", "Stage3", "stage3_model_final.pth"),
        os.path.join(DATA_PATH, "Model", "Stage3", "stage3_model_epoch_300.pth"),
        os.path.join(DATA_PATH, "stage3_model_epoch_10.pth"),
    ]
    loaded_s3 = False
    for ckpt in s3_checkpoints:
        if os.path.exists(ckpt):
            model_s3 = load_checkpoint(model_s3, ckpt, device)
            print(f"Loaded Stage 3 checkpoint: {ckpt}")
            loaded_s3 = True
            model_s3.freeze_reconstructor()
            break
    if not loaded_s3:
        print("WARNING: No Stage 3 checkpoint found, using Stage 2 weights")
        s2_state = model_s2.state_dict()
        model_s3.load_state_dict(s2_state, strict=False)
        model_s3.freeze_reconstructor()

    # =====================================================================
    # Multi-Stage Inference
    # =====================================================================
    print("\n" + "=" * 40)
    print("RUNNING MULTI-STAGE INFERENCE")
    print("=" * 40)

    models_dict = {
        'stage1': model_s1,
        'stage2': model_s2,
        'stage3': model_s3,
    }

    all_results, all_predictions = run_multistage_inference(
        models_dict, graph_data, device)

    # =====================================================================
    # Compute Metrics per Stage
    # =====================================================================
    print("\n" + "=" * 40)
    print("MULTI-STAGE COMPARISON RESULTS")
    print("=" * 40)

    stage_metrics = {}
    for stage_name in ['stage1', 'stage2', 'stage3']:
        preds = all_predictions[stage_name]
        if not preds:
            print(f"  {stage_name}: No predictions")
            continue
        true_vals = np.array([p[0] for p in preds])
        pred_vals = np.array([p[1] for p in preds])
        metrics = evaluate_metrics(true_vals, pred_vals)
        stage_metrics[stage_name] = metrics
        print(f"\n  {stage_name.upper()}:")
        print(f"    Pearson R: {metrics['Pearson']:.4f}")
        print(f"    RMSE:      {metrics['RMSE']:.4f} pKd")
        print(f"    MAE:       {metrics['MAE']:.4f} pKd")
        print(f"    SD:        {metrics['SD']:.4f} pKd")

    # =====================================================================
    # Edge Reconstruction Analysis (Stage 2/3 only)
    # =====================================================================
    print("\n" + "=" * 40)
    print("EDGE RECONSTRUCTION ANALYSIS")
    print("=" * 40)

    edge_analysis = {}
    for stage_name in ['stage2', 'stage3']:
        results = all_results.get(stage_name, {})
        if not results:
            continue

        keep_ratios = []
        remove_ratios = []
        add_ratios = []
        for comp_id, res in results.items():
            interp = res.get('interpretability', {})
            if 'edge_keep_ratio' in interp:
                keep_ratios.append(interp['edge_keep_ratio'])
                remove_ratios.append(interp['edge_remove_ratio'])
                add_ratios.append(interp['edge_add_ratio'])

        if keep_ratios:
            edge_analysis[stage_name] = {
                'avg_keep': float(np.mean(keep_ratios)),
                'avg_remove': float(np.mean(remove_ratios)),
                'avg_add': float(np.mean(add_ratios)),
                'std_keep': float(np.std(keep_ratios)),
                'std_remove': float(np.std(remove_ratios)),
                'std_add': float(np.std(add_ratios)),
            }
            print(f"\n  {stage_name.upper()} Edge Classification:")
            print(f"    Keep:   {np.mean(keep_ratios)*100:.1f}% ± {np.std(keep_ratios)*100:.1f}%")
            print(f"    Remove: {np.mean(remove_ratios)*100:.1f}% ± {np.std(remove_ratios)*100:.1f}%")
            print(f"    Add:    {np.mean(add_ratios)*100:.1f}% ± {np.std(add_ratios)*100:.1f}%")

    # =====================================================================
    # Affinity Stratification per Stage
    # =====================================================================
    print("\n" + "=" * 40)
    print("AFFINITY-STRATIFIED PERFORMANCE")
    print("=" * 40)

    stratification = {}
    for stage_name in ['stage1', 'stage2', 'stage3']:
        preds = all_predictions.get(stage_name, [])
        if not preds:
            continue

        strat = {'high': [], 'medium': [], 'low': []}
        for true_val, pred_val, cid in preds:
            error = pred_val - true_val
            if true_val >= 9.0:
                strat['high'].append(error)
            elif true_val >= 7.0:
                strat['medium'].append(error)
            else:
                strat['low'].append(error)

        strat_metrics = {}
        for group, errors in strat.items():
            if errors:
                strat_metrics[group] = {
                    'count': len(errors),
                    'mae': float(np.mean(np.abs(errors))),
                    'rmse': float(np.sqrt(np.mean(np.array(errors)**2))),
                    'mean_error': float(np.mean(errors)),
                }
        stratification[stage_name] = strat_metrics

        print(f"\n  {stage_name.upper()}:")
        for group in ['high', 'medium', 'low']:
            if group in strat_metrics:
                m = strat_metrics[group]
                print(f"    {group.capitalize()} (n={m['count']}): "
                      f"MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")

    # =====================================================================
    # Key Interpretability Features per Stage
    # =====================================================================
    print("\n" + "=" * 40)
    print("INTERPRETABILITY FEATURE SUMMARY")
    print("=" * 40)

    feature_summary = {}
    for stage_name in ['stage1', 'stage2', 'stage3']:
        results = all_results.get(stage_name, {})
        if not results:
            continue

        from scipy import stats as sp_stats
        comp_ids = sorted(results.keys())
        true_pkd = np.array([results[c]['true_pKd'] for c in comp_ids])

        feature_corrs = {}
        interp_keys = list(results[comp_ids[0]]['interpretability'].keys())
        for key in interp_keys:
            if key == 'complex_id':
                continue
            vals = np.array([results[c]['interpretability'].get(key, float('nan'))
                            for c in comp_ids])
            mask = ~np.isnan(vals)
            if mask.sum() < 3:
                continue
            r_val, p_val = sp_stats.pearsonr(vals[mask], true_pkd[mask])
            feature_corrs[key] = {'r': float(r_val), 'p': float(p_val)}

        # Top 5 features by absolute correlation
        top_features = sorted(feature_corrs.items(),
                             key=lambda x: abs(x[1]['r']), reverse=True)[:5]
        feature_summary[stage_name] = {
            'top_features': [(k, v) for k, v in top_features]
        }

        print(f"\n  {stage_name.upper()} Top Features (correlated with pKd):")
        for name, val in top_features:
            print(f"    {name}: r = {val['r']:+.4f} (p = {val['p']:.2e})")

    # =====================================================================
    # Save comprehensive output
    # =====================================================================
    output = {
        'case_study': 'HIV-1 Protease Multi-Stage',
        'num_complexes': len(graph_data),
        'stage_metrics': stage_metrics,
        'edge_analysis': edge_analysis,
        'affinity_stratification': stratification,
        'feature_summary': feature_summary,
    }

    # Per-complex details for Stage 3 (best model)
    stage3_results = all_results.get('stage3', {})
    per_complex = {}
    for comp_id, res in sorted(stage3_results.items(),
                                key=lambda x: x[1]['true_pKd']):
        per_complex[comp_id] = {
            'true_pKd': float(res['true_pKd']),
            'predicted_pKd': float(res['predicted_pKd']),
            'error': float(res['error']),
            'graph_topology': res['graph_topology'],
            'interpretability': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in res['interpretability'].items()
            },
        }
    output['per_complex'] = per_complex

    # Stage comparison per complex
    stage_comparison = {}
    for comp_id in graph_data:
        comp = {}
        for stage_name in ['stage1', 'stage2', 'stage3']:
            if comp_id in all_results.get(stage_name, {}):
                r = all_results[stage_name][comp_id]
                comp[stage_name] = {
                    'predicted_pKd': float(r['predicted_pKd']),
                    'error': float(r['error']),
                }
        if len(comp) == 3:
            stage_comparison[comp_id] = comp
    output['stage_comparison'] = stage_comparison

    # Save
    output_path = os.path.join(SERVER_BASE, "case_study", "multistage_case_study_report.json")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nFull report saved to {output_path}")

    # Save all representations
    output_pkl = os.path.join(SERVER_BASE, "case_study", "multistage_representations.pkl")
    with open(output_pkl, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Representations saved to {output_pkl}")

    print("\n" + "=" * 60)
    print("MULTI-STAGE CASE STUDY COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
