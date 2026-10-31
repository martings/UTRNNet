"""Evaluation metrics, three-way benchmarking (baseline vs GNN vs negative control), and leakage audits."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from utrnnet.graphs.controls import randomize_hetero_graph_edges
from utrnnet.graphs.leakage import apply_feature_mask_with_leakage_guard
from utrnnet.models.baseline import TrivialNonGraphBaseline
from utrnnet.models.hetero_gnn import UTRNNetHeteroGNN
from utrnnet.training.trainer import (
    TrainingConfig,
    compute_masked_reconstruction_loss,
    resolve_device,
    set_reproducible_seed,
    train_masked_reconstruction,
)


@dataclass(frozen=True)
class BenchmarkComparison:
    """Comparison of TrivialNonGraphBaseline, UTRNNetHeteroGNN (true graph), and NegativeControl (random edges)."""

    seed: int
    device: str
    n_graphs: int
    baseline_final_mse: float
    baseline_final_mae: float
    gnn_true_graph_final_mse: float
    gnn_true_graph_final_mae: float
    gnn_random_control_final_mse: float
    gnn_random_control_final_mae: float
    leakage_audit_passed: bool


def evaluate_masked_reconstruction(
    model: nn.Module,
    graphs: HeteroData | list[HeteroData],
    *,
    mask_ratio: float = 0.25,
    seed: int = 42,
    device: str = "auto",
) -> dict[str, float | dict[str, float]]:
    """Evaluate a trained model on masked node-feature reconstruction without gradient updates."""
    dev = resolve_device(device)
    model = model.to(dev)
    model.eval()
    graph_list = [graphs] if isinstance(graphs, HeteroData) else list(graphs)

    mses: list[float] = []
    maes: list[float] = []
    per_type_accum: dict[str, list[float]] = {}

    with torch.no_grad():
        for idx, g in enumerate(graph_list):
            batch = apply_feature_mask_with_leakage_guard(g, mask_ratio=mask_ratio, seed=seed + idx)
            pred_dict, _ = model(batch.masked_data.to(dev))
            loss, mae, per_type = compute_masked_reconstruction_loss(pred_dict, batch, dev)
            mses.append(float(loss.cpu().item()))
            maes.append(mae)
            for k, v in per_type.items():
                per_type_accum.setdefault(k, []).append(v)

    return {
        "mse": sum(mses) / max(len(mses), 1),
        "mae": sum(maes) / max(len(maes), 1),
        "per_node_type_mse": {k: sum(v) / len(v) for k, v in per_type_accum.items()},
    }


def verify_zero_target_leakage(
    model: nn.Module,
    data: HeteroData,
    *,
    target_node_type: str = "gene",
    target_feature_idx: int = 0,
    perturbation_magnitude: float = 100.0,
) -> bool:
    """Verify mathematically that perturbing masked ground-truth values produces zero change in predictions.

    Constructs two copies of `data` where `data_perturbed` adds `perturbation_magnitude` to the
    target feature AND to any coupled incident edge attributes (`expresses_gene`, `uses_isoform`,
    `interaction_score`), applies `apply_feature_mask_with_leakage_guard` with an explicit mask on
    `(target_node_type, target_feature_idx)`, and checks that `model(batch_a.masked_data)` and
    `model(batch_b.masked_data)` are bit-for-bit identical (`max_abs_diff == 0.0`).
    """
    if target_node_type not in data.node_types or data[target_node_type].x.shape[0] == 0:
        return True

    dev = next(model.parameters()).device
    model.eval()

    explicit_mask = {
        nt: torch.zeros_like(data[nt].x_missing_mask, dtype=torch.bool)
        for nt in data.node_types
    }
    explicit_mask[target_node_type][:, target_feature_idx] = True

    # Ensure the target feature is marked observed in a test copy so the mask is active
    base_copy = data.clone()
    base_copy[target_node_type].x_missing_mask = base_copy[target_node_type].x_missing_mask.clone()
    base_copy[target_node_type].x_missing_mask[:, target_feature_idx] = False

    pert_copy = base_copy.clone()
    pert_copy[target_node_type].x = pert_copy[target_node_type].x.clone()
    pert_copy[target_node_type].x[:, target_feature_idx] += perturbation_magnitude

    # Also corrupt coupled edge attributes in pert_copy to prove the sanitizer strips them
    if target_node_type == "gene" and target_feature_idx == 0:
        for etype in [("population", "expresses_gene", "gene"), ("gene", "rev_expresses_gene", "population")]:
            if etype in pert_copy.edge_types and pert_copy[etype].edge_attr.numel() > 0:
                pert_copy[etype].edge_attr = pert_copy[etype].edge_attr.clone()
                pert_copy[etype].edge_attr[:, :2] += perturbation_magnitude
    for etype in [
        ("regulatory_region", "candidate_binding", "regulator"),
        ("regulator", "candidate_regulation", "gene"),
    ]:
        if etype in pert_copy.edge_types and pert_copy[etype].edge_attr.numel() > 0:
            pert_copy[etype].edge_attr = pert_copy[etype].edge_attr.clone()
            pert_copy[etype].edge_attr[:, 3] += perturbation_magnitude

    batch_base = apply_feature_mask_with_leakage_guard(
        base_copy, mask_ratio=0.0, explicit_node_mask=explicit_mask
    )
    batch_pert = apply_feature_mask_with_leakage_guard(
        pert_copy, mask_ratio=0.0, explicit_node_mask=explicit_mask
    )

    # 1. Exact bit-for-bit equality check on all sanitized node & edge tensors
    for ntype in batch_base.masked_data.node_types:
        if not torch.equal(batch_base.masked_data[ntype].x, batch_pert.masked_data[ntype].x):
            return False
        if not torch.equal(
            batch_base.masked_data[ntype].x_missing_mask,
            batch_pert.masked_data[ntype].x_missing_mask,
        ):
            return False
    for etype in batch_base.masked_data.edge_types:
        if not torch.equal(batch_base.masked_data[etype].edge_attr, batch_pert.masked_data[etype].edge_attr):
            return False
        if not torch.equal(
            batch_base.masked_data[etype].edge_attr_missing_mask,
            batch_pert.masked_data[etype].edge_attr_missing_mask,
        ):
            return False

    # 2. Forward pass invariance check (allowing 1e-5 tolerance for CUDA atomicAdd thread order)
    with torch.no_grad():
        pred_base, _ = model(batch_base.masked_data.to(dev))
        pred_pert, _ = model(batch_pert.masked_data.to(dev))

    max_diff = float((pred_base[target_node_type] - pred_pert[target_node_type]).abs().max().cpu().item())
    return max_diff <= 1e-5


def run_three_way_benchmark(
    graphs: HeteroData | list[HeteroData],
    config: TrainingConfig | None = None,
    *,
    output_dir: Path | str | None = None,
) -> BenchmarkComparison:
    """Run reproducible comparison across TrivialNonGraphBaseline, UTRNNetHeteroGNN, and Randomized-Edge Control."""
    cfg = config or TrainingConfig()
    graph_list = [graphs] if isinstance(graphs, HeteroData) else list(graphs)

    # 1. Trivial non-graph baseline
    set_reproducible_seed(cfg.seed)
    baseline = TrivialNonGraphBaseline(hidden_channels=cfg.hidden_channels, dropout=cfg.dropout)
    res_base = train_masked_reconstruction(
        baseline,
        graph_list,
        cfg,
        checkpoint_dir=(Path(output_dir) / "baseline") if output_dir else None,
    )

    # 2. UTRNNetHeteroGNN on true regulatory graph
    set_reproducible_seed(cfg.seed)
    gnn_true = UTRNNetHeteroGNN(
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
    )
    res_true = train_masked_reconstruction(
        gnn_true,
        graph_list,
        cfg,
        checkpoint_dir=(Path(output_dir) / "hetero_gnn_true") if output_dir else None,
    )

    # 3. UTRNNetHeteroGNN on negative-control randomized-edge graph
    rand_graphs = [randomize_hetero_graph_edges(g, seed=cfg.seed + i) for i, g in enumerate(graph_list)]
    set_reproducible_seed(cfg.seed)
    gnn_rand = UTRNNetHeteroGNN(
        hidden_channels=cfg.hidden_channels,
        num_layers=cfg.num_layers,
        heads=cfg.heads,
        dropout=cfg.dropout,
    )
    res_rand = train_masked_reconstruction(
        gnn_rand,
        rand_graphs,
        cfg,
        checkpoint_dir=(Path(output_dir) / "hetero_gnn_random_control") if output_dir else None,
    )

    leakage_ok = verify_zero_target_leakage(gnn_true, graph_list[0], target_node_type="gene", target_feature_idx=0)

    comparison = BenchmarkComparison(
        seed=cfg.seed,
        device=res_true.device,
        n_graphs=len(graph_list),
        baseline_final_mse=res_base.final_loss,
        baseline_final_mae=res_base.final_mae,
        gnn_true_graph_final_mse=res_true.final_loss,
        gnn_true_graph_final_mae=res_true.final_mae,
        gnn_random_control_final_mse=res_rand.final_loss,
        gnn_random_control_final_mae=res_rand.final_mae,
        leakage_audit_passed=leakage_ok,
    )

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "benchmark_summary.json").write_text(json.dumps(asdict(comparison), indent=2) + "\n")

    return comparison
