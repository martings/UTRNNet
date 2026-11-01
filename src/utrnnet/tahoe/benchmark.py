"""Four-way Tahoe-100M perturbation benchmark with identical splits and 95% bootstrap uncertainty (`Task C`)."""
from __future__ import annotations

import resource
import time
from typing import Any
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from utrnnet.tahoe.adapter import validate_group_partition
from utrnnet.tahoe.models import (
    GeneGraphDeltaPredictor,
    RegularizedMLPDeltaPredictor,
    RegulatoryPriorHeteroDeltaPredictor,
    RidgeDeltaPredictor,
    zero_delta,
)


def held_out_condition_mask(conditions: list[str], held_out: set[str]) -> np.ndarray:
    if not held_out:
        raise ValueError("at least one complete condition must be held out")
    condition_array = np.asarray(conditions, dtype=str)
    if not held_out.issubset(set(condition_array)):
        raise ValueError("held-out condition absent from input")
    return np.isin(condition_array, sorted(held_out))


def response_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    *,
    top_k: int = 10,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute MSE, MAE, Pearson r, directional sign agreement, top-k DE recovery, and 95% bootstrap CIs."""
    y = np.asarray(truth, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("truth and prediction must be aligned finite 2D arrays")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    mse = float(np.mean((y - p) ** 2))
    mae = float(np.mean(np.abs(y - p)))

    direction_mask = np.abs(y) > 1e-6
    direction = (
        float(np.mean(np.sign(y[direction_mask]) == np.sign(p[direction_mask])))
        if direction_mask.any()
        else float("nan")
    )

    # Sample-wise Pearson r across genes
    pearsons: list[float] = []
    for i in range(y.shape[0]):
        yi, pi = y[i], p[i]
        if np.std(yi) > 1e-8 and np.std(pi) > 1e-8:
            pearsons.append(float(np.corrcoef(yi, pi)[0, 1]))
        else:
            pearsons.append(0.0)
    mean_pearson = float(np.mean(pearsons)) if pearsons else 0.0

    k = min(top_k, y.shape[1])
    true_top = np.argpartition(np.abs(y), -k, axis=1)[:, -k:]
    pred_top = np.argpartition(np.abs(p), -k, axis=1)[:, -k:]
    recovery = (
        float(np.mean([len(set(a.tolist()) & set(b.tolist())) / k for a, b in zip(true_top, pred_top)]))
        if len(y)
        else float("nan")
    )

    # Bootstrap 95% confidence intervals across sample-line rows
    rng = np.random.default_rng(seed)
    n_rows = y.shape[0]
    if n_rows >= 2 and n_bootstrap > 0:
        row_mses = np.mean((y - p) ** 2, axis=1)
        row_maes = np.mean(np.abs(y - p), axis=1)
        boot_idx = rng.integers(0, n_rows, size=(n_bootstrap, n_rows))
        boot_mse = row_mses[boot_idx].mean(axis=1)
        boot_mae = row_maes[boot_idx].mean(axis=1)
        mse_ci95 = [float(np.percentile(boot_mse, 2.5)), float(np.percentile(boot_mse, 97.5))]
        mae_ci95 = [float(np.percentile(boot_mae, 2.5)), float(np.percentile(boot_mae, 97.5))]
    else:
        mse_ci95 = [mse, mse]
        mae_ci95 = [mae, mae]

    return {
        "mse": mse,
        "mse_ci95": mse_ci95,
        "mae": mae,
        "mae_ci95": mae_ci95,
        "mean_pearson_r": mean_pearson,
        "directional_agreement_nonzero_truth": direction,
        "top_k_de_recovery": recovery,
        "top_k": k,
        "n_samples": int(y.shape[0]),
        "n_genes": int(y.shape[1]),
    }


def run_four_model_tahoe_benchmark(
    *,
    sample_ids: list[str],
    conditions: list[str],
    basal_log_counts: np.ndarray,
    context_features: np.ndarray,
    delta_targets: np.ndarray,
    gene_edge_index: np.ndarray,
    reg_to_gene_edge_index: np.ndarray,
    reg_to_gene_edge_weight: np.ndarray,
    gene_structural_priors: np.ndarray,
    n_regulators: int,
    train_mask: np.ndarray,
    ridge_alpha: float = 1.0,
    latent_dim: int = 64,
    epochs: int = 60,
    lr: float = 5e-3,
    top_k: int = 10,
    seed: int = 42,
    device: str = "cpu",
) -> dict[str, Any]:
    """Compare all 4 required Tahoe delta predictors on an identical sample-isolated train/test partition.

    Models compared:
    1. `zero_delta`: Zero-delta baseline (`delta = 0`).
    2. `ridge` & `mlp`: Regularized linear (Ridge) and regularized MLP baselines.
    3. `gene_graph_gnn`: Homogeneous gene-level graph predictor (`GeneGraphDeltaPredictor`).
    4. `regulatory_prior_hetero_gnn`: Heterogeneous regulatory-prior GNN (`RegulatoryPriorHeteroDeltaPredictor`).
    """
    t0 = time.perf_counter()
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = ~train_mask
    if int(train_mask.sum()) < 2 or int(test_mask.sum()) < 1:
        raise ValueError("Benchmark split requires >= 2 training rows and >= 1 test row")

    # Enforce strict sample-level partition isolation (`validate_group_partition`)
    part_df = pd.DataFrame(
        {
            "sample": sample_ids,
            "condition": conditions,
            "partition": ["train" if m else "test" for m in train_mask],
        }
    )
    validate_group_partition(part_df, group_column="sample", partition_column="partition")

    x_basal = np.asarray(basal_log_counts, dtype=np.float32)
    x_ctx = np.asarray(context_features, dtype=np.float32)
    y_delta = np.asarray(delta_targets, dtype=np.float32)
    x_concat = np.concatenate([x_basal, x_ctx], axis=1)

    n_genes = x_basal.shape[1]
    ctx_dim = x_ctx.shape[1]

    # 1. Zero-delta baseline
    pred_zero = zero_delta(int(test_mask.sum()), n_genes)

    # 2a. Regularized linear (Ridge) baseline
    ridge = RidgeDeltaPredictor(alpha=ridge_alpha).fit(x_concat[train_mask], y_delta[train_mask])
    pred_ridge = ridge.predict(x_concat[test_mask])

    # Convert tensors for PyTorch models
    torch.manual_seed(seed)
    np.random.seed(seed)

    b_tr = torch.tensor(x_basal[train_mask], dtype=torch.float32, device=dev)
    c_tr = torch.tensor(x_ctx[train_mask], dtype=torch.float32, device=dev)
    y_tr = torch.tensor(y_delta[train_mask], dtype=torch.float32, device=dev)

    b_te = torch.tensor(x_basal[test_mask], dtype=torch.float32, device=dev)
    c_te = torch.tensor(x_ctx[test_mask], dtype=torch.float32, device=dev)

    g_ei = torch.tensor(gene_edge_index, dtype=torch.long, device=dev)
    rg_ei = torch.tensor(reg_to_gene_edge_index, dtype=torch.long, device=dev)
    rg_ew = torch.tensor(reg_to_gene_edge_weight, dtype=torch.float32, device=dev)
    g_priors = torch.tensor(gene_structural_priors, dtype=torch.float32, device=dev)

    # 2b. Regularized MLP baseline
    torch.manual_seed(seed)
    mlp = RegularizedMLPDeltaPredictor(n_genes=n_genes, context_dim=ctx_dim, latent_dim=latent_dim).to(dev)
    opt_mlp = torch.optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-2)
    mlp.train()
    for _ in range(epochs):
        opt_mlp.zero_grad()
        loss = F.mse_loss(mlp(b_tr, c_tr), y_tr)
        loss.backward()
        opt_mlp.step()
    mlp.eval()
    with torch.no_grad():
        pred_mlp = mlp(b_te, c_te).detach().cpu().numpy()

    # 3. Gene-level graph predictor
    torch.manual_seed(seed)
    gene_gnn = GeneGraphDeltaPredictor(n_genes=n_genes, context_dim=ctx_dim, latent_dim=latent_dim, layers=2).to(dev)
    opt_gg = torch.optim.AdamW(gene_gnn.parameters(), lr=lr, weight_decay=1e-2)
    gene_gnn.train()
    for _ in range(epochs):
        opt_gg.zero_grad()
        loss = F.mse_loss(gene_gnn(b_tr, c_tr, g_ei), y_tr)
        loss.backward()
        opt_gg.step()
    gene_gnn.eval()
    with torch.no_grad():
        pred_gene_gnn = gene_gnn(b_te, c_te, g_ei).detach().cpu().numpy()

    # 4. Regulatory-prior heterogeneous GNN
    torch.manual_seed(seed)
    hetero_gnn = RegulatoryPriorHeteroDeltaPredictor(
        n_genes=n_genes,
        n_regulators=max(n_regulators, 1),
        context_dim=ctx_dim,
        structural_prior_dim=g_priors.shape[1],
        latent_dim=latent_dim,
        layers=2,
    ).to(dev)
    opt_hg = torch.optim.AdamW(hetero_gnn.parameters(), lr=lr, weight_decay=1e-2)
    hetero_gnn.train()
    for _ in range(epochs):
        opt_hg.zero_grad()
        out = hetero_gnn(b_tr, c_tr, g_ei, rg_ei, rg_ew, g_priors)
        loss = F.mse_loss(out, y_tr)
        loss.backward()
        opt_hg.step()
    hetero_gnn.eval()
    with torch.no_grad():
        pred_hetero_gnn = hetero_gnn(b_te, c_te, g_ei, rg_ei, rg_ew, g_priors).detach().cpu().numpy()

    y_te = y_delta[test_mask]
    elapsed = time.perf_counter() - t0
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    peak_vram_bytes = (
        int(torch.cuda.max_memory_allocated(dev))
        if (dev.type == "cuda" and torch.cuda.is_available())
        else 0
    )

    return {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "n_genes": int(n_genes),
        "n_regulators": int(n_regulators),
        "sample_partition_isolation_verified": True,
        "missing_modalities": ["PAS usage", "timepoint"],
        "models": {
            "zero_delta": response_metrics(y_te, pred_zero, top_k=top_k, seed=seed),
            "ridge": response_metrics(y_te, pred_ridge, top_k=top_k, seed=seed),
            "regularized_mlp": response_metrics(y_te, pred_mlp, top_k=top_k, seed=seed),
            "gene_graph_gnn": response_metrics(y_te, pred_gene_gnn, top_k=top_k, seed=seed),
            "regulatory_prior_hetero_gnn": response_metrics(y_te, pred_hetero_gnn, top_k=top_k, seed=seed),
        },
        "resources": {
            "elapsed_seconds": round(elapsed, 4),
            "observed_peak_ram_bytes": peak_rss_bytes,
            "observed_peak_vram_bytes": peak_vram_bytes,
        },
    }
