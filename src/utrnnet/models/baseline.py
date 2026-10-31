"""Trivial non-graph baseline for masked node-feature reconstruction."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from utrnnet.contracts.schema import NODE_FEATURE_SCHEMAS, NODE_TYPES


class TrivialNonGraphBaseline(nn.Module):
    """Node-local MLP baseline that strictly ignores all graph edges.

    Maps each node's observed feature vector and observedness mask `[x * (~mask), (~mask)]`
    to a latent representation of dimension `hidden_channels` (default 64) and decodes back
    to the node feature space. Serves as the lower-bound reference against message passing.
    """

    def __init__(self, hidden_channels: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.dropout = float(dropout)

        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        for ntype in NODE_TYPES:
            in_dim = len(NODE_FEATURE_SCHEMAS[ntype])
            self.encoders[ntype] = nn.Sequential(
                nn.Linear(in_dim * 2, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.GELU(),
            )
            self.decoders[ntype] = nn.Linear(self.hidden_channels, in_dim)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Compute edge-free latent representations per node type."""
        z_dict: dict[str, torch.Tensor] = {}
        for ntype in data.node_types:
            if ntype not in self.encoders:
                continue
            x = data[ntype].x
            mask = data[ntype].x_missing_mask
            if x.shape[0] == 0:
                z_dict[ntype] = torch.zeros((0, self.hidden_channels), dtype=x.dtype, device=x.device)
                continue
            obs = (~mask).to(x.dtype)
            x_clean = x * obs
            inp = torch.cat([x_clean, obs], dim=-1)
            z_dict[ntype] = self.encoders[ntype](inp)
        return z_dict

    def decode(self, z_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Decode latent node states back to original feature dimensions."""
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
        """Return `(pred_dict, z_dict)`."""
        z_dict = self.encode(data)
        pred_dict = self.decode(z_dict)
        return pred_dict, z_dict
