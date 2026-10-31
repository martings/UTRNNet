"""Target-feature masking and strict anti-leakage sanitization for UTRNNet."""
from __future__ import annotations

from dataclasses import dataclass
import torch
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class MaskedGraphBatch:
    """Container holding a masked/sanitized `HeteroData` input and ground-truth reconstruction targets."""

    masked_data: HeteroData
    target_values: dict[str, torch.Tensor]
    reconstruction_mask: dict[str, torch.Tensor]
    sanitized_edge_types: list[tuple[str, str, str]]


def strip_target_expression_leakage(
    data: HeteroData,
    *,
    mask_gene_expression: bool = True,
    mask_isoform_usage_in_scores: bool = True,
) -> tuple[HeteroData, list[tuple[str, str, str]]]:
    """Remove any edge features that incorporate target gene expression or target isoform usage."""
    clean = data.clone()
    sanitized: list[tuple[str, str, str]] = []

    if mask_gene_expression:
        for etype in [
            ("population", "expresses_gene", "gene"),
            ("gene", "rev_expresses_gene", "population"),
        ]:
            if etype in clean.edge_types and clean[etype].edge_attr.numel() > 0:
                clean[etype].edge_attr = clean[etype].edge_attr.clone()
                clean[etype].edge_attr_missing_mask = clean[etype].edge_attr_missing_mask.clone()
                clean[etype].edge_attr[:, :2] = 0.0
                clean[etype].edge_attr_missing_mask[:, :2] = True
                sanitized.append(etype)

    if mask_isoform_usage_in_scores:
        for etype in [
            ("regulatory_region", "candidate_binding", "regulator"),
            ("regulator", "rev_candidate_binding", "regulatory_region"),
            ("regulator", "candidate_regulation", "gene"),
            ("gene", "rev_candidate_regulation", "regulator"),
        ]:
            if etype in clean.edge_types and clean[etype].edge_attr.numel() > 0:
                clean[etype].edge_attr = clean[etype].edge_attr.clone()
                clean[etype].edge_attr_missing_mask = clean[etype].edge_attr_missing_mask.clone()
                # Column 3 is interaction_score (which incorporates relative_isoform_usage and abundance_factor)
                clean[etype].edge_attr[:, 3] = 0.0
                clean[etype].edge_attr_missing_mask[:, 3] = True
                sanitized.append(etype)

    return clean, sanitized


def apply_feature_mask_with_leakage_guard(
    data: HeteroData,
    *,
    mask_ratio: float = 0.25,
    seed: int = 42,
    target_node_types: tuple[str, ...] = ("gene", "pas", "utr_isoform", "regulatory_region", "regulator"),
    explicit_node_mask: dict[str, torch.Tensor] | None = None,
) -> MaskedGraphBatch:
    """Mask observed node features for reconstruction and sanitize any coupled node/edge attributes.

    Guarantees zero leakage of masked target values through:
    1. Incident `expresses_gene` edges when `gene` expression (col 0) is masked.
    2. Incident `uses_isoform` edges and coupled `pas` <-> `utr_isoform` usage features when PAS/isoform
       usage (col 2/3) is masked.
    3. Incident `expresses_regulator` edges and coupled `regulator` `log1p_mean_umi_per_cell` (col 0) <->
       `abundance_factor` (col 1) when regulator abundance is masked.
    4. Candidate `interaction_score` (col 3 on `candidate_binding` and `candidate_regulation` edges), which
       is always stripped during masked reconstruction so derived scores never leak usage or abundance.
    """
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio must be in [0, 1], got {mask_ratio}")

    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    masked_data = data.clone()
    target_values: dict[str, torch.Tensor] = {}
    reconstruction_mask: dict[str, torch.Tensor] = {}

    for ntype in data.node_types:
        store = data[ntype]
        x_orig = store.x.detach().cpu().clone()
        miss_orig = store.x_missing_mask.detach().cpu().clone()
        target_values[ntype] = x_orig

        if ntype not in target_node_types or x_orig.numel() == 0:
            reconstruction_mask[ntype] = torch.zeros_like(miss_orig, dtype=torch.bool)
            continue

        observable = ~miss_orig
        if explicit_node_mask is not None and ntype in explicit_node_mask:
            rmask = explicit_node_mask[ntype].detach().cpu().to(torch.bool) & observable
        else:
            rand_u = torch.rand(x_orig.shape, generator=gen)
            rmask = (rand_u < mask_ratio) & observable
            if mask_ratio > 0.0 and observable.any() and not rmask.any():
                obs_coords = torch.nonzero(observable, as_tuple=False)
                pick = int(torch.randint(0, obs_coords.shape[0], (1,), generator=gen).item())
                r_i, r_j = int(obs_coords[pick, 0]), int(obs_coords[pick, 1])
                rmask[r_i, r_j] = True

        reconstruction_mask[ntype] = rmask
        if rmask.any():
            new_x = masked_data[ntype].x.clone()
            new_m = masked_data[ntype].x_missing_mask.clone()
            rmask_dev = rmask.to(new_x.device)
            new_x[rmask_dev] = 0.0
            new_m[rmask_dev] = True
            masked_data[ntype].x = new_x
            masked_data[ntype].x_missing_mask = new_m

    # Always strip composite interaction_score from candidate edges during masked feature reconstruction
    masked_data, sanitized_edges = strip_target_expression_leakage(
        masked_data,
        mask_gene_expression=False,
        mask_isoform_usage_in_scores=True,
    )

    # 1. Gene expression leakage guard (gene col 0 <-> expresses_gene edges)
    if "gene" in reconstruction_mask and reconstruction_mask["gene"].numel() > 0:
        masked_genes = reconstruction_mask["gene"][:, 0]
        if bool(masked_genes.any()):
            for etype, gene_end in [
                (("population", "expresses_gene", "gene"), 1),
                (("gene", "rev_expresses_gene", "population"), 0),
            ]:
                if etype in masked_data.edge_types and masked_data[etype].edge_index.numel() > 0:
                    ei = masked_data[etype].edge_index
                    hit = masked_genes.to(ei.device)[ei[gene_end]]
                    if bool(hit.any()):
                        masked_data[etype].edge_attr = masked_data[etype].edge_attr.clone()
                        masked_data[etype].edge_attr_missing_mask = masked_data[etype].edge_attr_missing_mask.clone()
                        masked_data[etype].edge_attr[hit, :2] = 0.0
                        masked_data[etype].edge_attr_missing_mask[hit, :2] = True
                        sanitized_edges.append(etype)

    # 2. Regulator abundance leakage guard (regulator col 0 <-> col 1 <-> expresses_regulator edges)
    if "regulator" in reconstruction_mask and reconstruction_mask["regulator"].numel() > 0:
        reg_rmask = reconstruction_mask["regulator"]
        masked_regs = reg_rmask[:, 0] | reg_rmask[:, 1]
        if bool(masked_regs.any()):
            dev = masked_data["regulator"].x.device
            m_dev = masked_regs.to(dev)
            masked_data["regulator"].x = masked_data["regulator"].x.clone()
            masked_data["regulator"].x_missing_mask = masked_data["regulator"].x_missing_mask.clone()
            masked_data["regulator"].x[m_dev, :2] = 0.0
            masked_data["regulator"].x_missing_mask[m_dev, :2] = True
            for etype, reg_end in [
                (("population", "expresses_regulator", "regulator"), 1),
                (("regulator", "rev_expresses_regulator", "population"), 0),
            ]:
                if etype in masked_data.edge_types and masked_data[etype].edge_index.numel() > 0:
                    ei = masked_data[etype].edge_index
                    hit = m_dev[ei[reg_end]]
                    if bool(hit.any()):
                        masked_data[etype].edge_attr = masked_data[etype].edge_attr.clone()
                        masked_data[etype].edge_attr_missing_mask = masked_data[etype].edge_attr_missing_mask.clone()
                        masked_data[etype].edge_attr[hit, :2] = 0.0
                        masked_data[etype].edge_attr_missing_mask[hit, :2] = True
                        sanitized_edges.append(etype)

    # 3. PAS & UTR isoform usage leakage guard (pas cols 2,3,4 <-> utr_isoform col 2 <-> uses_isoform edges)
    pas_masked = (
        (reconstruction_mask["pas"][:, 2] | reconstruction_mask["pas"][:, 3] | reconstruction_mask["pas"][:, 4])
        if ("pas" in reconstruction_mask and reconstruction_mask["pas"].numel() > 0)
        else torch.zeros((0,), dtype=torch.bool)
    )
    iso_masked = (
        reconstruction_mask["utr_isoform"][:, 2]
        if ("utr_isoform" in reconstruction_mask and reconstruction_mask["utr_isoform"].numel() > 0)
        else torch.zeros((0,), dtype=torch.bool)
    )
    if bool(pas_masked.any()) or bool(iso_masked.any()):
        # Propagate masking across ("pas", "defines_isoform", "utr_isoform")
        def_etype = ("pas", "defines_isoform", "utr_isoform")
        if def_etype in masked_data.edge_types and masked_data[def_etype].edge_index.numel() > 0:
            ei = masked_data[def_etype].edge_index.detach().cpu()
            for p_i, u_i in zip(ei[0].tolist(), ei[1].tolist()):
                if pas_masked.numel() > p_i and pas_masked[p_i] and iso_masked.numel() > u_i:
                    iso_masked[u_i] = True
                if iso_masked.numel() > u_i and iso_masked[u_i] and pas_masked.numel() > p_i:
                    pas_masked[p_i] = True

        if pas_masked.numel() > 0 and bool(pas_masked.any()):
            dev = masked_data["pas"].x.device
            pm_dev = pas_masked.to(dev)
            masked_data["pas"].x = masked_data["pas"].x.clone()
            masked_data["pas"].x_missing_mask = masked_data["pas"].x_missing_mask.clone()
            masked_data["pas"].x[pm_dev, 2:5] = 0.0
            masked_data["pas"].x_missing_mask[pm_dev, 2:5] = True

        if iso_masked.numel() > 0 and bool(iso_masked.any()):
            dev = masked_data["utr_isoform"].x.device
            im_dev = iso_masked.to(dev)
            masked_data["utr_isoform"].x = masked_data["utr_isoform"].x.clone()
            masked_data["utr_isoform"].x_missing_mask = masked_data["utr_isoform"].x_missing_mask.clone()
            masked_data["utr_isoform"].x[im_dev, 2] = 0.0
            masked_data["utr_isoform"].x_missing_mask[im_dev, 2] = True

            for etype, iso_end in [
                (("population", "uses_isoform", "utr_isoform"), 1),
                (("utr_isoform", "rev_uses_isoform", "population"), 0),
            ]:
                if etype in masked_data.edge_types and masked_data[etype].edge_index.numel() > 0:
                    ei = masked_data[etype].edge_index
                    hit = im_dev[ei[iso_end]]
                    if bool(hit.any()):
                        masked_data[etype].edge_attr = masked_data[etype].edge_attr.clone()
                        masked_data[etype].edge_attr_missing_mask = masked_data[etype].edge_attr_missing_mask.clone()
                        masked_data[etype].edge_attr[hit, :3] = 0.0
                        masked_data[etype].edge_attr_missing_mask[hit, :3] = True
                        sanitized_edges.append(etype)

    return MaskedGraphBatch(
        masked_data=masked_data,
        target_values=target_values,
        reconstruction_mask=reconstruction_mask,
        sanitized_edge_types=sanitized_edges,
    )
