"""Negative-control graph transformations (edge randomization preserving node features and edge counts)."""
from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from utrnnet.contracts.schema import validate_cell_regulatory_state


def randomize_hetero_graph_edges(data: HeteroData, *, seed: int = 42) -> HeteroData:
    """Create a negative-control `HeteroData` graph by randomly rewiring edges within each relation type.

    Preserves:
    - All node features `x` and `x_missing_mask`
    - Number of edges per relation `(src_type, rel, dst_type)`
    - Edge attributes `edge_attr`, `edge_attr_missing_mask`, and `evidence_tier_id` (permuted across edges)
    - Bidirectional consistency between `(src, rel, dst)` and `(dst, rev_rel, src)`
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    ctrl = data.clone()
    forward_etypes = [et for et in ctrl.edge_types if not et[1].startswith("rev_")]

    for etype in forward_etypes:
        src_t, rel, dst_t = etype
        store = ctrl[etype]
        n_edges = int(store.edge_index.shape[1])
        n_src = int(ctrl[src_t].x.shape[0])
        n_dst = int(ctrl[dst_t].x.shape[0])

        if n_edges == 0 or n_src == 0 or n_dst == 0:
            continue

        dev = store.edge_index.device
        rand_src = torch.randint(0, n_src, (n_edges,), generator=gen, dtype=torch.long).to(dev)
        rand_dst = torch.randint(0, n_dst, (n_edges,), generator=gen, dtype=torch.long).to(dev)
        attr_perm = torch.randperm(n_edges, generator=gen, dtype=torch.long).to(dev)

        new_ei = torch.stack([rand_src, rand_dst], dim=0)
        new_ea = store.edge_attr[attr_perm].clone()
        new_em = store.edge_attr_missing_mask[attr_perm].clone()
        new_ti = store.evidence_tier_id[attr_perm].clone()
        perm_cpu = attr_perm.detach().cpu().tolist()
        old_labels = list(store.evidence_tier_labels)
        new_labels = [old_labels[i] for i in perm_cpu]

        store.edge_index = new_ei
        store.edge_attr = new_ea
        store.edge_attr_missing_mask = new_em
        store.evidence_tier_id = new_ti
        store.evidence_tier_labels = new_labels

        rev_etype = (dst_t, f"rev_{rel}", src_t)
        if rev_etype in ctrl.edge_types:
            rstore = ctrl[rev_etype]
            rstore.edge_index = torch.stack([rand_dst, rand_src], dim=0)
            rstore.edge_attr = new_ea.clone()
            rstore.edge_attr_missing_mask = new_em.clone()
            rstore.evidence_tier_id = new_ti.clone()
            rstore.evidence_tier_labels = list(new_labels)

    ctrl.is_negative_control = True
    ctrl.negative_control_seed = int(seed)
    validate_cell_regulatory_state(ctrl, raise_on_error=True)
    return ctrl
