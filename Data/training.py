"""Canonical three-stage training pipeline for the paper implementation."""

from __future__ import annotations

import json
import math
import pickle
import random

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .MMDCG_DTA_Stage1 import MMDCGDTAModel_Stage1
from .MMDCG_DTA_Stage2 import MMDCGDTAModel_Stage2
from .MMDCG_DTA_Stage3 import MMDCGDTAModel_Stage3


GRAPH_KEYS = (
    "ligand_atom_graph",
    "protein_atom_graph",
    "atom_interaction_graph",
    "atom_candidate_graph",
    "ligand_fragment_graph",
    "protein_residue_graph",
    "substructure_interaction_graph",
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_sample(sample):
    missing = [key for key in GRAPH_KEYS if key not in sample]
    if missing:
        raise ValueError(
            f"graph cache is missing {missing}; rebuild it with "
            "`python -m Data.build_graph_dataset`"
        )
    for key in ("ligand_atom_graph", "protein_atom_graph"):
        if "group" not in sample[key].ndata:
            raise ValueError(
                "graph cache lacks exact atom-to-substructure IDs; rebuild the cache"
            )


def collate_samples(samples):
    samples = [
        sample
        for sample in samples
        if sample is not None and sample.get("label") is not None
    ]
    if not samples:
        return None
    for sample in samples:
        validate_sample(sample)
    batch = {key: dgl.batch([sample[key] for sample in samples]) for key in GRAPH_KEYS}
    batch["label"] = torch.tensor(
        [float(sample["label"]) for sample in samples], dtype=torch.float32
    ).reshape(-1, 1)
    batch["compound_id"] = [sample.get("compound_id", "unknown") for sample in samples]
    return batch


def move_batch(batch, device):
    return {
        key: value.to(device) if key in GRAPH_KEYS or torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _load_graph_dictionary(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; generate paper-aligned caches with "
            "`python -m Data.build_graph_dataset`"
        )
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return data


def make_loaders(config, data_dir):
    refined = _load_graph_dictionary(data_dir / "refined_set_graphs.pkl")
    core = _load_graph_dictionary(data_dir / "core_set_graphs.pkl")
    core_ids = set(core)
    # CASF Core complexes are evaluation-only even if present in the Refined cache.
    train_pool = [sample for key, sample in refined.items() if key not in core_ids]
    if len(train_pool) < 2:
        raise ValueError("at least two non-Core training samples are required")
    rng = random.Random(config["seed"])
    rng.shuffle(train_pool)
    validation_size = max(1, int(round(len(train_pool) * config["validation_ratio"])))
    validation = train_pool[:validation_size]
    training = train_pool[validation_size:]
    test = list(core.values())

    loader_options = dict(
        batch_size=config["batch_size"],
        collate_fn=collate_samples,
        num_workers=config.get("num_workers", 0),
    )
    return {
        "train": DataLoader(training, shuffle=True, **loader_options),
        "validation": DataLoader(validation, shuffle=False, **loader_options),
        "test": DataLoader(test, shuffle=False, **loader_options),
    }


def _prediction(model, batch):
    output = model(batch)
    return output[0] if isinstance(output, tuple) else output


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    rmse = float(np.sqrt(np.mean((targets - predictions) ** 2)))
    mae = float(np.mean(np.abs(targets - predictions)))
    pcc = (
        float(np.corrcoef(targets, predictions)[0, 1])
        if len(targets) > 1
        else float("nan")
    )
    if len(targets) > 1 and np.std(predictions) > 0:
        slope, intercept = np.polyfit(predictions, targets, 1)
        residuals = targets - (slope * predictions + intercept)
        sd = float(np.sqrt(np.sum(residuals**2) / (len(targets) - 1)))
    else:
        sd = float("nan")
    return {"RMSE": rmse, "MAE": mae, "SD": sd, "PCC": pcc}


def evaluate(model, loader, device):
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            batch = move_batch(batch, device)
            prediction = _prediction(model, batch)
            targets.extend(batch["label"].detach().cpu().reshape(-1).tolist())
            predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
    if not targets:
        raise ValueError("evaluation loader yielded no valid samples")
    return regression_metrics(targets, predictions)


def _save_checkpoint(model, path, stage, epoch, validation_metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "epoch": epoch,
            "validation_metrics": validation_metrics,
            "model_state": model.state_dict(),
        },
        path,
    )


def _load_checkpoint(model, path, device, strict):
    if not path.exists():
        raise FileNotFoundError(f"required checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device)
    state = payload.get("model_state", payload)
    return model.load_state_dict(state, strict=strict)


def _train_affinity_stage(
    model, loaders, device, learning_rate, epochs, patience, checkpoint, stage
):
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    loss_function = nn.MSELoss()
    best_validation = math.inf
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in loaders["train"]:
            if batch is None:
                continue
            batch = move_batch(batch, device)
            optimizer.zero_grad()
            loss = loss_function(_prediction(model, batch), batch["label"])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            batches += 1
        validation = evaluate(model, loaders["validation"], device)
        print(
            f"Stage {stage} epoch {epoch}: train MSE={running_loss / max(batches, 1):.6f}, "
            f"validation RMSE={validation['RMSE']:.6f}"
        )
        if validation["RMSE"] < best_validation:
            best_validation = validation["RMSE"]
            stale_epochs = 0
            _save_checkpoint(model, checkpoint, stage, epoch, validation)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    _load_checkpoint(model, checkpoint, device, strict=True)
    return model


def train_stage1(config, loaders, device, output_dir):
    model = MMDCGDTAModel_Stage1(config).to(device)
    return _train_affinity_stage(
        model,
        loaders,
        device,
        config["stage1_learning_rate"],
        config["stage1_epochs"],
        config["patience"],
        output_dir / "stage1_best.pt",
        stage=1,
    )


def _edge_targets(graph):
    distances = graph.edata["dist"].reshape(-1)
    labels = torch.zeros_like(distances, dtype=torch.long)
    labels[(distances >= 3.5) & (distances < 6.0)] = 1
    labels[distances < 3.5] = 2
    return labels


def _combined_edge_loss(auxiliary, batch, loss_function):
    logits = []
    labels = []
    graph_by_hierarchy = {
        "atom": batch["atom_candidate_graph"],
        "substructure": batch["substructure_interaction_graph"],
    }
    for hierarchy, graph in graph_by_hierarchy.items():
        hierarchy_logits = auxiliary[hierarchy]["logits"]
        if hierarchy_logits.numel():
            logits.append(hierarchy_logits)
            labels.append(_edge_targets(graph))
    if not logits:
        return None, None
    joined_logits = torch.cat(logits, dim=0)
    joined_labels = torch.cat(labels, dim=0)
    return loss_function(joined_logits, joined_labels), joined_logits.argmax(dim=-1)


def _set_stage2_trainable(model, reconstructors):
    reconstructor_parameters = {
        id(parameter)
        for reconstructor in model.reconstructors
        for parameter in reconstructor.parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad = (
            id(parameter) in reconstructor_parameters
        ) == reconstructors


def train_stage2(config, loaders, device, output_dir):
    model = MMDCGDTAModel_Stage2(config).to(device)
    stage1_checkpoint = output_dir / "stage1_best.pt"
    _load_checkpoint(model, stage1_checkpoint, device, strict=False)
    edge_optimizer = torch.optim.Adam(
        [
            parameter
            for reconstructor in model.reconstructors
            for parameter in reconstructor.parameters()
        ],
        lr=config["stage2_learning_rate"],
    )
    main_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "edge_reconstructor" not in name
    ]
    main_optimizer = torch.optim.Adam(
        main_parameters, lr=config["stage2_learning_rate"]
    )
    edge_loss_function = nn.CrossEntropyLoss()
    affinity_loss_function = nn.MSELoss()
    checkpoint = output_dir / "stage2_best.pt"
    best_validation = math.inf
    stale_epochs = 0

    for epoch in range(1, config["stage2_epochs"] + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for batch in loaders["train"]:
            if batch is None:
                continue
            batch = move_batch(batch, device)
            previous_classes = None
            _set_stage2_trainable(model, reconstructors=True)
            for _inner_step in range(config["inner_max_iterations"]):
                edge_optimizer.zero_grad()
                _prediction_value, auxiliary = model(batch)
                edge_loss, classes = _combined_edge_loss(
                    auxiliary, batch, edge_loss_function
                )
                if edge_loss is None:
                    break
                edge_loss.backward()
                edge_optimizer.step()
                if previous_classes is not None:
                    change_ratio = (classes != previous_classes).float().mean().item()
                    if change_ratio <= config["inner_tolerance"]:
                        break
                previous_classes = classes.detach()

            _set_stage2_trainable(model, reconstructors=False)
            main_optimizer.zero_grad()
            prediction, _auxiliary = model(batch)
            affinity_loss = affinity_loss_function(prediction, batch["label"])
            affinity_loss.backward()
            main_optimizer.step()
            running_loss += affinity_loss.item()
            batches += 1

        for parameter in model.parameters():
            parameter.requires_grad = True
        validation = evaluate(model, loaders["validation"], device)
        print(
            f"Stage 2 epoch {epoch}: train MSE={running_loss / max(batches, 1):.6f}, "
            f"validation RMSE={validation['RMSE']:.6f}"
        )
        if validation["RMSE"] < best_validation:
            best_validation = validation["RMSE"]
            stale_epochs = 0
            _save_checkpoint(model, checkpoint, 2, epoch, validation)
        else:
            stale_epochs += 1
            if stale_epochs >= config["patience"]:
                break
    _load_checkpoint(model, checkpoint, device, strict=True)
    return model


def train_stage3(config, loaders, device, output_dir):
    model = MMDCGDTAModel_Stage3(config).to(device)
    _load_checkpoint(model, output_dir / "stage2_best.pt", device, strict=True)
    model.freeze_reconstructors()
    return _train_affinity_stage(
        model,
        loaders,
        device,
        config["stage3_learning_rate"],
        config["stage3_epochs"],
        config["patience"],
        output_dir / "stage3_best.pt",
        stage=3,
    )


def run_pipeline(config, data_dir, output_dir, requested_stage="all"):
    seed_everything(config["seed"])
    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    loaders = make_loaders(config, data_dir)
    if requested_stage in ("all", "1"):
        train_stage1(config, loaders, device, output_dir)
    if requested_stage in ("all", "2"):
        train_stage2(config, loaders, device, output_dir)
    if requested_stage in ("all", "3"):
        model = train_stage3(config, loaders, device, output_dir)
        test_metrics = evaluate(model, loaders["test"], device)
        (output_dir / "core_test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2) + "\n"
        )
        print(f"Held-out Core test metrics: {test_metrics}")
        return test_metrics
    return None
