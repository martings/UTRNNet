"""Two-layer heterogeneous message-passing encoder and reconstruction/perturbation heads for UTRNNet."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import TransformerConv

from utrnnet.contracts.schema import (
    CANONICAL_EDGE_TYPES,
    EDGE_CATEGORIES,
    EDGE_FEATURE_SCHEMAS,
    EVIDENCE_TIER_TO_ID,
    NODE_FEATURE_SCHEMAS,
    NODE_TYPES,
)


def _edge_key(etype: tuple[str, str, str]) -> str:
    return f"{etype[0]}__{etype[1]}__{etype[2]}"


class TypedHeteroMessagePassingLayer(nn.Module):
    """Single heterogeneous message-passing layer preserving edge categories, evidence tiers, and missingness."""

    def __init__(
        self,
        hidden_channels: int = 64,
        heads: int = 4,
        dropout: float = 0.1,
        allowed_edge_categories: tuple[str, ...] = EDGE_CATEGORIES,
        use_evidence_tiers: bool = True,
    ) -> None:
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError(f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})")
        self.hidden_channels = int(hidden_channels)
        self.heads = int(heads)
        self.allowed_edge_categories = set(allowed_edge_categories)
        self.use_evidence_tiers = bool(use_evidence_tiers)

        self.evidence_emb = nn.Embedding(len(EVIDENCE_TIER_TO_ID) + 2, self.hidden_channels)
        self.edge_proj = nn.ModuleDict()
        self.convs = nn.ModuleDict()

        for etype, cat in CANONICAL_EDGE_TYPES.items():
            if cat not in self.allowed_edge_categories:
                continue
            ekey = _edge_key(etype)
            efeat_dim = len(EDGE_FEATURE_SCHEMAS[cat])
            self.edge_proj[ekey] = nn.Sequential(
                nn.Linear(efeat_dim * 2 + self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
            )
            self.convs[ekey] = TransformerConv(
                in_channels=(self.hidden_channels, self.hidden_channels),
                out_channels=self.hidden_channels // self.heads,
                heads=self.heads,
                concat=True,
                beta=True,
                dropout=dropout,
                edge_dim=self.hidden_channels,
            )

        self.norms = nn.ModuleDict({nt: nn.LayerNorm(self.hidden_channels) for nt in NODE_TYPES})
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        z_dict: dict[str, torch.Tensor],
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:
        agg_messages: dict[str, list[torch.Tensor]] = {nt: [] for nt in z_dict}

        for etype in data.edge_types:
            ekey = _edge_key(etype)
            if ekey not in self.convs:
                continue
            src_t, _, dst_t = etype
            if src_t not in z_dict or dst_t not in z_dict:
                continue
            estore = data[etype]
            if getattr(estore, "edge_category", None) not in self.allowed_edge_categories:
                continue
            ei = estore.edge_index
            if ei.numel() == 0 or z_dict[src_t].shape[0] == 0 or z_dict[dst_t].shape[0] == 0:
                continue

            ea = estore.edge_attr
            em = estore.edge_attr_missing_mask
            obs_e = (~em).to(ea.dtype)
            ea_clean = ea * obs_e
            if not self.use_evidence_tiers:
                # Ablation mode: replace evidence_factor column with uniform 1.0 and zero evidence embedding
                ea_clean = ea_clean.clone()
                cat = getattr(estore, "edge_category", "")
                if cat in EDGE_FEATURE_SCHEMAS and "evidence_factor" in EDGE_FEATURE_SCHEMAS[cat]:
                    ev_col = EDGE_FEATURE_SCHEMAS[cat].index("evidence_factor")
                    ea_clean[:, ev_col] = 1.0
                ev_vec = torch.zeros((ei.shape[1], self.hidden_channels), dtype=ea.dtype, device=ea.device)
            else:
                ev_ids = estore.evidence_tier_id.clamp(min=0, max=len(EVIDENCE_TIER_TO_ID) + 1)
                ev_vec = self.evidence_emb(ev_ids)
            edge_emb = self.edge_proj[ekey](torch.cat([ea_clean, obs_e, ev_vec], dim=-1))

            msg = self.convs[ekey]((z_dict[src_t], z_dict[dst_t]), ei, edge_attr=edge_emb)
            agg_messages[dst_t].append(msg)

        out_dict: dict[str, torch.Tensor] = {}
        for ntype, z in z_dict.items():
            if z.shape[0] == 0 or not agg_messages[ntype]:
                out_dict[ntype] = z
            else:
                stacked = torch.stack(agg_messages[ntype], dim=0).sum(dim=0)
                out_dict[ntype] = self.norms[ntype](z + self.drop(self.act(stacked)))
        return out_dict


class PerturbationConditionedHead(nn.Module):
    """Scaffold head for predicting post-perturbation transcriptomic/APA state deltas from regulator perturbations.

    Note: Kept strictly as a discrete perturbation-conditioning scaffold (not a continuous-time ODE),
    pending experimental RBP knockdown/overexpression validation cohorts.
    """

    def __init__(self, hidden_channels: int = 64) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.pert_proj = nn.Linear(1, self.hidden_channels)
        self.gene_delta_head = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, len(NODE_FEATURE_SCHEMAS["gene"])),
        )
        self.iso_delta_head = nn.Sequential(
            nn.Linear(self.hidden_channels * 2, self.hidden_channels),
            nn.GELU(),
            nn.Linear(self.hidden_channels, len(NODE_FEATURE_SCHEMAS["utr_isoform"])),
        )

    def forward(
        self,
        z_dict: dict[str, torch.Tensor],
        data: HeteroData,
        regulator_perturbation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict feature deltas `{'gene': delta_gene, 'utr_isoform': delta_iso}` under `regulator_perturbation`."""
        z_reg = z_dict["regulator"]
        pert_vec = self.pert_proj(regulator_perturbation.reshape(-1, 1))
        reg_shifted = z_reg + pert_vec

        # Aggregate regulator perturbation context onto genes via ("regulator", "candidate_regulation", "gene")
        z_gene = z_dict["gene"]
        gene_ctx = torch.zeros_like(z_gene)
        etype_rg = ("regulator", "candidate_regulation", "gene")
        if etype_rg in data.edge_types and data[etype_rg].edge_index.numel() > 0:
            ei = data[etype_rg].edge_index
            gene_ctx.index_add_(0, ei[1], reg_shifted[ei[0]] - z_reg[ei[0]])

        # Aggregate onto isoforms via region -> isoform if available
        z_iso = z_dict["utr_isoform"]
        iso_ctx = torch.zeros_like(z_iso)
        etype_rb = ("regulator", "rev_candidate_binding", "regulatory_region")
        etype_ri = ("regulatory_region", "rev_contains_site", "utr_isoform")
        if (
            etype_rb in data.edge_types
            and etype_ri in data.edge_types
            and data[etype_rb].edge_index.numel() > 0
            and data[etype_ri].edge_index.numel() > 0
        ):
            reg_delta = reg_shifted - z_reg
            region_ctx = torch.zeros_like(z_dict["regulatory_region"])
            ei_rb = data[etype_rb].edge_index
            region_ctx.index_add_(0, ei_rb[1], reg_delta[ei_rb[0]])
            ei_ri = data[etype_ri].edge_index
            iso_ctx.index_add_(0, ei_ri[1], region_ctx[ei_ri[0]])

        return {
            "gene": self.gene_delta_head(torch.cat([z_gene, gene_ctx], dim=-1)),
            "utr_isoform": self.iso_delta_head(torch.cat([z_iso, iso_ctx], dim=-1)),
        }


class UTRNNetHeteroGNN(nn.Module):
    """Two-layer heterogeneous message-passing encoder for `cell-regulatory-state/v0.1` graphs."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
        allowed_edge_categories: tuple[str, ...] = EDGE_CATEGORIES,
        use_evidence_tiers: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.num_layers = int(num_layers)
        self.heads = int(heads)
        self.dropout = float(dropout)
        self.allowed_edge_categories = tuple(allowed_edge_categories)
        self.use_evidence_tiers = bool(use_evidence_tiers)

        self.input_proj = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for ntype in NODE_TYPES:
            in_dim = len(NODE_FEATURE_SCHEMAS[ntype])
            self.input_proj[ntype] = nn.Sequential(
                nn.Linear(in_dim * 2, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
            )
            self.decoders[ntype] = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
                nn.Linear(self.hidden_channels, in_dim),
            )

        self.layers = nn.ModuleList(
            [
                TypedHeteroMessagePassingLayer(
                    hidden_channels=self.hidden_channels,
                    heads=self.heads,
                    dropout=self.dropout,
                    allowed_edge_categories=self.allowed_edge_categories,
                    use_evidence_tiers=self.use_evidence_tiers,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.perturbation_head = PerturbationConditionedHead(hidden_channels=self.hidden_channels)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode node features and typed regulatory topology into `hidden_channels` embeddings."""
        z_dict: dict[str, torch.Tensor] = {}
        for ntype in data.node_types:
            if ntype not in self.input_proj:
                continue
            x = data[ntype].x
            mask = data[ntype].x_missing_mask
            if x.shape[0] == 0:
                z_dict[ntype] = torch.zeros((0, self.hidden_channels), dtype=x.dtype, device=x.device)
                continue
            obs = (~mask).to(x.dtype)
            x_clean = x * obs
            z_dict[ntype] = self.input_proj[ntype](torch.cat([x_clean, obs], dim=-1))

        for layer in self.layers:
            z_dict = layer(z_dict, data)
        return z_dict

    def decode(self, z_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode latent embeddings back to node feature space for masked feature reconstruction."""
        pred_dict: dict[str, torch.Tensor] = {}
        for ntype, z in z_dict.items():
            if ntype not in self.decoders:
                continue
            if z.shape[0] == 0:
                out_dim = len(NODE_FEATURE_SCHEMAS[ntype])
                pred_dict[ntype] = torch.zeros((0, out_dim), dtype=z.dtype, device=z.device)
            else:
                pred_dict[ntype] = self.decoders[ntype](z)
        return pred_dict

    def forward(self, data: HeteroData) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Execute forward pass returning `(reconstructed_features_dict, latent_embeddings_dict)`."""
        z_dict = self.encode(data)
        pred_dict = self.decode(z_dict)
        return pred_dict, z_dict

    def predict_perturbation_delta(
        self,
        data: HeteroData,
        regulator_perturbation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict post-perturbation state deltas from a regulator perturbation vector."""
        z_dict = self.encode(data)
        return self.perturbation_head(z_dict, data, regulator_perturbation)
