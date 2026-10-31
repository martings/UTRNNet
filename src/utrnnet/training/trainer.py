"""Reproducible masked node-feature reconstruction training and checkpointing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from utrnnet.contracts.schema import SCHEMA_VERSION
from utrnnet.graphs.leakage import MaskedGraphBatch, apply_feature_mask_with_leakage_guard


def set_reproducible_seed(seed: int = 42) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch (CPU and CUDA)."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve `'auto'`, `'cpu'`, or `'cuda'` to a `torch.device`."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class TrainingConfig:
    """Configuration for masked node-feature reconstruction training."""

    seed: int = 42
    hidden_channels: int = 64
    num_layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-3
    weight_decay: float = 1e-4
    epochs: int = 30
    mask_ratio: float = 0.25
    device: str = "auto"
    target_node_types: tuple[str, ...] = (
        "gene",
        "pas",
        "utr_isoform",
        "regulatory_region",
        "regulator",
    )


@dataclass
class TrainingResult:
    """Summary of a masked node-feature reconstruction training run."""

    model_type: str
    device: str
    seed: int
    epochs: int
    initial_loss: float
    final_loss: float
    final_mae: float
    per_node_type_mse: dict[str, float]
    loss_history: list[float] = field(default_factory=list)
    checkpoint_path: str | None = None
    metrics_path: str | None = None


def compute_masked_reconstruction_loss(
    pred_dict: dict[str, torch.Tensor],
    batch: MaskedGraphBatch,
    device: torch.device,
) -> tuple[torch.Tensor, float, dict[str, float]]:
    """Compute MSE and MAE strictly over masked, originally observed node features."""
    sq_errors: list[torch.Tensor] = []
    abs_errors: list[torch.Tensor] = []
    per_type_mse: dict[str, float] = {}

    for ntype, rmask_cpu in batch.reconstruction_mask.items():
        if ntype not in pred_dict or not bool(rmask_cpu.any()):
            continue
        pred = pred_dict[ntype]
        rmask = rmask_cpu.to(device)
        target = batch.target_values[ntype].to(device)
        diff = pred[rmask] - target[rmask]
        sq = diff.pow(2)
        ab = diff.abs()
        sq_errors.append(sq)
        abs_errors.append(ab)
        per_type_mse[ntype] = float(sq.detach().mean().cpu().item())

    if not sq_errors:
        zero = torch.tensor(0.0, device=device, requires_grad=True)
        return zero, 0.0, per_type_mse

    all_sq = torch.cat(sq_errors, dim=0)
    all_ab = torch.cat(abs_errors, dim=0)
    loss = all_sq.mean()
    mae = float(all_ab.detach().mean().cpu().item())
    return loss, mae, per_type_mse


def train_masked_reconstruction(
    model: nn.Module,
    graphs: HeteroData | list[HeteroData],
    config: TrainingConfig | None = None,
    *,
    checkpoint_dir: Path | str | None = None,
    fixed_mask_across_epochs: bool = True,
) -> TrainingResult:
    """Train a model on masked node-feature reconstruction with anti-leakage sanitization."""
    cfg = config or TrainingConfig()
    set_reproducible_seed(cfg.seed)
    dev = resolve_device(cfg.device)
    model = model.to(dev)

    graph_list = [graphs] if isinstance(graphs, HeteroData) else list(graphs)
    if not graph_list:
        raise ValueError("Cannot train on an empty list of graphs")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # Pre-build deterministic masked batches if fixed_mask_across_epochs=True
    fixed_batches: list[MaskedGraphBatch] = []
    fixed_gpu_graphs: list[HeteroData] = []
    if fixed_mask_across_epochs:
        for idx, g in enumerate(graph_list):
            b = apply_feature_mask_with_leakage_guard(
                g,
                mask_ratio=cfg.mask_ratio,
                seed=cfg.seed + idx,
                target_node_types=cfg.target_node_types,
            )
            fixed_batches.append(b)
            fixed_gpu_graphs.append(b.masked_data.to(dev))

    loss_history: list[float] = []
    final_mae = 0.0
    final_per_type: dict[str, float] = {}

    model.train()
    for epoch in range(cfg.epochs):
        optimizer.zero_grad()
        epoch_losses: list[torch.Tensor] = []
        epoch_maes: list[float] = []
        type_accum: dict[str, list[float]] = {}

        for idx, g in enumerate(graph_list):
            if fixed_mask_across_epochs:
                batch = fixed_batches[idx]
                m_data = fixed_gpu_graphs[idx]
            else:
                batch = apply_feature_mask_with_leakage_guard(
                    g,
                    mask_ratio=cfg.mask_ratio,
                    seed=cfg.seed + epoch * 1000 + idx,
                    target_node_types=cfg.target_node_types,
                )
                m_data = batch.masked_data.to(dev)
            pred_dict, _ = model(m_data)
            loss, mae, per_type = compute_masked_reconstruction_loss(pred_dict, batch, dev)
            epoch_losses.append(loss)
            epoch_maes.append(mae)
            for k, v in per_type.items():
                type_accum.setdefault(k, []).append(v)

        total_loss = torch.stack(epoch_losses).mean()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        loss_val = float(total_loss.detach().cpu().item())
        loss_history.append(loss_val)
        final_mae = float(np.mean(epoch_maes))
        final_per_type = {k: float(np.mean(vals)) for k, vals in type_accum.items()}

    ckpt_path_str: str | None = None
    metrics_path_str: str | None = None
    if checkpoint_dir is not None:
        cdir = Path(checkpoint_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        ckpt_file = cdir / "checkpoint.pt"
        metrics_file = cdir / "metrics.json"
        torch.save(
            {
                "schema_version": SCHEMA_VERSION,
                "model_class": model.__class__.__name__,
                "model_state_dict": model.state_dict(),
                "config": asdict(cfg),
                "loss_history": loss_history,
                "final_loss": loss_history[-1] if loss_history else 0.0,
                "final_mae": final_mae,
                "per_node_type_mse": final_per_type,
            },
            ckpt_file,
        )
        ckpt_path_str = str(ckpt_file)
        metrics_payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_type": model.__class__.__name__,
            "device": str(dev),
            "seed": cfg.seed,
            "epochs": cfg.epochs,
            "initial_loss": loss_history[0] if loss_history else 0.0,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "final_mae": final_mae,
            "per_node_type_mse": final_per_type,
            "loss_history": loss_history,
            "scientific_disclaimer": (
                "Metrics reflect masked feature reconstruction on candidate regulatory graphs; "
                "not molecular occupancy or longitudinal causal dynamics."
            ),
        }
        metrics_file.write_text(json.dumps(metrics_payload, indent=2) + "\n")
        metrics_path_str = str(metrics_file)

    return TrainingResult(
        model_type=model.__class__.__name__,
        device=str(dev),
        seed=cfg.seed,
        epochs=cfg.epochs,
        initial_loss=loss_history[0] if loss_history else 0.0,
        final_loss=loss_history[-1] if loss_history else 0.0,
        final_mae=final_mae,
        per_node_type_mse=final_per_type,
        loss_history=loss_history,
        checkpoint_path=ckpt_path_str,
        metrics_path=metrics_path_str,
    )
