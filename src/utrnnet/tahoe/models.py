"""Four fixed-endpoint delta-expression model classes for the Tahoe-100M perturbation benchmark (`Task C`)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def zero_delta(n_samples: int, n_genes: int) -> np.ndarray:
    """Model 1: Zero-delta baseline predicting zero change from matched DMSO_TF control."""
    if n_samples < 0 or n_genes < 0:
        raise ValueError("output dimensions must be nonnegative")
    return np.zeros((n_samples, n_genes), dtype=np.float32)


@dataclass
class RidgeDeltaPredictor:
    """Model 2a: Multi-output regularized linear (Ridge) predictor over basal + drug/dose/cell-line context."""

    alpha: float = 1.0
    coefficients_: np.ndarray | None = None
    intercept_: np.ndarray | None = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> "RidgeDeltaPredictor":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
            raise ValueError("features and targets must be aligned 2D arrays with at least two rows")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("training arrays must contain only finite values")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        self._x_mean = x.mean(axis=0)
        self.intercept_ = y.mean(axis=0)
        centered_x = x - self._x_mean
        centered_y = y - self.intercept_
        gram = centered_x @ centered_x.T
        dual = np.linalg.solve(gram + self.alpha * np.eye(len(x)), centered_y)
        self.coefficients_ = centered_x.T @ dual
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.coefficients_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coefficients_.shape[0]:
            raise ValueError("features have the wrong shape")
        return ((x - self._x_mean) @ self.coefficients_ + self.intercept_).astype(np.float32)


class RegularizedMLPDeltaPredictor(nn.Module):
    """Model 2b: Regularized non-graph MLP predicting gene delta from basal expression and perturbation context."""

    def __init__(self, n_genes: int, context_dim: int, latent_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_genes = int(n_genes)
        self.context_dim = int(context_dim)
        self.net = nn.Sequential(
            nn.Linear(n_genes + context_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_genes),
        )

    def forward(self, basal_log_counts: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([basal_log_counts, context], dim=-1))


class GeneGraphDeltaPredictor(nn.Module):
    """Model 3: Homogeneous gene-level graph predictor over basal expression, context, and gene-gene edges."""

    def __init__(self, n_genes: int, context_dim: int, latent_dim: int = 64, layers: int = 2) -> None:
        super().__init__()
        if n_genes < 1 or context_dim < 1 or latent_dim < 1 or layers < 1:
            raise ValueError("model dimensions and layer count must be positive")
        self.n_genes = int(n_genes)
        self.gene_embedding = nn.Embedding(n_genes, latent_dim)
        self.expression_projection = nn.Linear(1, latent_dim)
        self.context_projection = nn.Linear(context_dim, latent_dim)
        self.self_layers = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(layers))
        self.neighbor_layers = nn.ModuleList(nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(latent_dim) for _ in range(layers))
        self.output = nn.Linear(latent_dim, 1)

    def forward(
        self,
        basal_log_counts: torch.Tensor,
        context: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        if basal_log_counts.ndim != 2 or basal_log_counts.shape[1] != self.n_genes:
            raise ValueError("basal_log_counts must have shape [batch, n_genes]")
        if context.ndim != 2 or context.shape[0] != basal_log_counts.shape[0]:
            raise ValueError("context must have shape [batch, context_features]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, n_edges]")

        gene_ix = torch.arange(self.n_genes, device=basal_log_counts.device)
        nodes = self.gene_embedding(gene_ix).unsqueeze(0)
        nodes = nodes + self.expression_projection(basal_log_counts.unsqueeze(-1))
        nodes = nodes + self.context_projection(context).unsqueeze(1)

        src, dst = edge_index[0].long(), edge_index[1].long()
        if src.numel() and (
            int(torch.max(torch.stack([src.max(), dst.max()]))) >= self.n_genes
            or int(torch.min(torch.stack([src.min(), dst.min()]))) < 0
        ):
            raise ValueError("edge index outside gene range")

        degree = torch.bincount(dst, minlength=self.n_genes).clamp_min(1).to(nodes.dtype).view(1, -1, 1)
        for self_layer, neighbor_layer, norm in zip(self.self_layers, self.neighbor_layers, self.norms):
            aggregate = torch.zeros_like(nodes)
            if src.numel():
                aggregate.index_add_(1, dst, nodes.index_select(1, src))
            nodes = norm(nodes + F.gelu(self_layer(nodes) + neighbor_layer(aggregate / degree)))
        return self.output(nodes).squeeze(-1)


class RegulatoryPriorHeteroDeltaPredictor(nn.Module):
    """Model 4: Heterogeneous regulatory-prior GNN coupling Tahoe gene states with UTRNet APA/RBP priors.

    Preserves epistemic separation:
    - Does NOT fabricate PAS usage or continuous time courses in Tahoe.
    - Consumes static UTRNet structural/sequence priors (`gene_structural_priors` and weighted
      `(regulator, candidate_regulation, gene)` edges) alongside Tahoe basal gene expression and
      perturbation context (`drug`, `log1p_dose`, `cell_line_id`, `target_gene_indicator`).
    """

    def __init__(
        self,
        n_genes: int,
        n_regulators: int,
        context_dim: int,
        structural_prior_dim: int = 4,
        latent_dim: int = 64,
        layers: int = 2,
    ) -> None:
        super().__init__()
        if min(n_genes, n_regulators, context_dim, structural_prior_dim, latent_dim, layers) < 1:
            raise ValueError("all model dimensions must be positive")
        self.n_genes = int(n_genes)
        self.n_regulators = int(n_regulators)
        self.latent_dim = int(latent_dim)
        self.layers = int(layers)

        self.gene_embedding = nn.Embedding(n_genes, latent_dim)
        self.reg_embedding = nn.Embedding(n_regulators, latent_dim)
        self.gene_expr_proj = nn.Linear(1 + structural_prior_dim, latent_dim)
        self.context_proj = nn.Linear(context_dim, latent_dim)
        self.reg_context_proj = nn.Linear(context_dim + n_genes, latent_dim)

        self.gene_self_layers = nn.ModuleList(nn.Linear(latent_dim, latent_dim) for _ in range(layers))
        self.gene_to_gene_layers = nn.ModuleList(nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(layers))
        self.reg_to_gene_layers = nn.ModuleList(nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(layers))
        self.gene_to_reg_layers = nn.ModuleList(nn.Linear(latent_dim, latent_dim, bias=False) for _ in range(layers))
        self.gene_norms = nn.ModuleList(nn.LayerNorm(latent_dim) for _ in range(layers))
        self.reg_norms = nn.ModuleList(nn.LayerNorm(latent_dim) for _ in range(layers))

        self.delta_head = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(
        self,
        basal_log_counts: torch.Tensor,
        context: torch.Tensor,
        gene_edge_index: torch.Tensor,
        reg_to_gene_edge_index: torch.Tensor,
        reg_to_gene_edge_weight: torch.Tensor,
        gene_structural_priors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = basal_log_counts.shape[0]
        dev = basal_log_counts.device

        gene_ix = torch.arange(self.n_genes, device=dev)
        reg_ix = torch.arange(self.n_regulators, device=dev)

        prior_expanded = gene_structural_priors.to(dev).unsqueeze(0).expand(batch_size, -1, -1)
        gene_in = torch.cat([basal_log_counts.unsqueeze(-1), prior_expanded], dim=-1)

        g_nodes = (
            self.gene_embedding(gene_ix).unsqueeze(0)
            + self.gene_expr_proj(gene_in)
            + self.context_proj(context).unsqueeze(1)
        )
        reg_global_ctx = self.reg_context_proj(torch.cat([context, basal_log_counts], dim=-1)).unsqueeze(1)
        r_nodes = self.reg_embedding(reg_ix).unsqueeze(0) + reg_global_ctx

        g_src, g_dst = gene_edge_index[0].long(), gene_edge_index[1].long()
        g_deg = torch.bincount(g_dst, minlength=self.n_genes).clamp_min(1).to(g_nodes.dtype).view(1, -1, 1)

        r_src, r_dst = reg_to_gene_edge_index[0].long(), reg_to_gene_edge_index[1].long()
        ew = reg_to_gene_edge_weight.to(g_nodes.dtype).view(1, -1, 1)
        rg_deg = torch.bincount(r_dst, minlength=self.n_genes).clamp_min(1).to(g_nodes.dtype).view(1, -1, 1)
        gr_deg = torch.bincount(r_src, minlength=self.n_regulators).clamp_min(1).to(g_nodes.dtype).view(1, -1, 1)

        g_init = g_nodes
        for l_idx in range(self.layers):
            gg_agg = torch.zeros_like(g_nodes)
            if g_src.numel() > 0:
                gg_agg.index_add_(1, g_dst, g_nodes.index_select(1, g_src))

            rg_agg = torch.zeros_like(g_nodes)
            gr_agg = torch.zeros_like(r_nodes)
            if r_src.numel() > 0:
                weighted_r_msgs = r_nodes.index_select(1, r_src) * ew
                rg_agg.index_add_(1, r_dst, weighted_r_msgs)
                weighted_g_msgs = g_nodes.index_select(1, r_dst) * ew
                gr_agg.index_add_(1, r_src, weighted_g_msgs)

            g_upd = (
                self.gene_self_layers[l_idx](g_nodes)
                + self.gene_to_gene_layers[l_idx](gg_agg / g_deg)
                + self.reg_to_gene_layers[l_idx](rg_agg / rg_deg)
            )
            g_nodes = self.gene_norms[l_idx](g_nodes + F.gelu(g_upd))
            r_nodes = self.reg_norms[l_idx](r_nodes + F.gelu(self.gene_to_reg_layers[l_idx](gr_agg / gr_deg)))

        return self.delta_head(torch.cat([g_nodes, g_init], dim=-1)).squeeze(-1)
