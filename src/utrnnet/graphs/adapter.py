"""Typed GraphML/CSV to PyTorch Geometric `HeteroData` adapter (`cell-regulatory-state/v0.1`)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from scpolaseq_utrnet.cis import (
    annotate_motif_survival_probabilities,
    compute_fractional_cis_dosage,
)
from utrnnet.contracts.schema import (
    AMBIGUITY_STATUS_TO_CODE,
    CANONICAL_EDGE_TYPES,
    CODE_TO_AMBIGUITY_STATUS,
    EDGE_FEATURE_SCHEMAS,
    EVIDENCE_TIER_FACTORS,
    EVIDENCE_TIER_TO_ID,
    NODE_FEATURE_SCHEMAS,
    NODE_TYPES,
    SCHEMA_VERSION,
    SPAN_STATUS_TO_CODE,
    validate_cell_regulatory_state,
)


def _safe_float(val: Any) -> tuple[float, bool]:
    """Return `(numeric_value, is_missing)` where NaN/None returns `(0.0, True)`."""
    if val is None:
        return 0.0, True
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return 0.0, True
        return f, False
    except (TypeError, ValueError):
        return 0.0, True


def _log1p_safe(val: Any) -> tuple[float, bool]:
    f, missing = _safe_float(val)
    if missing or f < 0.0:
        return 0.0, True
    return float(math.log1p(f)), False


def _init_empty_node_store(data: HeteroData, ntype: str) -> None:
    feat_names = NODE_FEATURE_SCHEMAS[ntype]
    n_feats = len(feat_names)
    data[ntype].x = torch.zeros((0, n_feats), dtype=torch.float32)
    data[ntype].x_missing_mask = torch.zeros((0, n_feats), dtype=torch.bool)
    data[ntype].feature_names = list(feat_names)
    data[ntype].node_ids = []
    data[ntype].node_id_to_index = {}
    if ntype == "utr_isoform":
        data[ntype].ambiguity_code = torch.zeros((0,), dtype=torch.long)
        data[ntype].ambiguity_labels = []
        data[ntype].span_status_code = torch.zeros((0,), dtype=torch.long)
        data[ntype].is_unresolved_mask = torch.zeros((0,), dtype=torch.bool)


def _set_node_store(
    data: HeteroData,
    ntype: str,
    node_ids: list[str],
    feats: list[list[float]],
    masks: list[list[bool]],
    extra: dict[str, Any] | None = None,
) -> dict[str, int]:
    feat_names = NODE_FEATURE_SCHEMAS[ntype]
    n_feats = len(feat_names)
    if not node_ids:
        _init_empty_node_store(data, ntype)
        return {}
    data[ntype].x = torch.tensor(feats, dtype=torch.float32).reshape(len(node_ids), n_feats)
    data[ntype].x_missing_mask = torch.tensor(masks, dtype=torch.bool).reshape(len(node_ids), n_feats)
    data[ntype].feature_names = list(feat_names)
    data[ntype].node_ids = list(node_ids)
    idx_map = {nid: i for i, nid in enumerate(node_ids)}
    data[ntype].node_id_to_index = idx_map
    if extra:
        for k, v in extra.items():
            setattr(data[ntype], k, v)
    return idx_map


def _set_edge_store(
    data: HeteroData,
    etype: tuple[str, str, str],
    src_indices: list[int],
    dst_indices: list[int],
    edge_feats: list[list[float]],
    edge_masks: list[list[bool]],
    evidence_tiers: list[str],
    uses_target_expression: bool = False,
) -> None:
    cat = CANONICAL_EDGE_TYPES[etype]
    feat_names = EDGE_FEATURE_SCHEMAS[cat]
    n_efeats = len(feat_names)
    n_edges = len(src_indices)
    store = data[etype]
    store.edge_category = cat
    store.feature_names = list(feat_names)
    store.uses_target_expression = bool(uses_target_expression)

    if n_edges == 0:
        store.edge_index = torch.zeros((2, 0), dtype=torch.long)
        store.edge_attr = torch.zeros((0, n_efeats), dtype=torch.float32)
        store.edge_attr_missing_mask = torch.zeros((0, n_efeats), dtype=torch.bool)
        store.evidence_tier_id = torch.zeros((0,), dtype=torch.long)
        store.evidence_tier_labels = []
        store.is_candidate_only = torch.zeros((0,), dtype=torch.bool)
        return

    store.edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    store.edge_attr = torch.tensor(edge_feats, dtype=torch.float32).reshape(n_edges, n_efeats)
    store.edge_attr_missing_mask = torch.tensor(edge_masks, dtype=torch.bool).reshape(n_edges, n_efeats)
    tier_ids = [EVIDENCE_TIER_TO_ID.get(t, 0) for t in evidence_tiers]
    store.evidence_tier_id = torch.tensor(tier_ids, dtype=torch.long)
    store.evidence_tier_labels = list(evidence_tiers)
    is_cand = cat == "inferred_candidate"
    store.is_candidate_only = torch.full((n_edges,), is_cand, dtype=torch.bool)


def _add_reverse_edges(data: HeteroData) -> None:
    """Create typed reverse edges for message passing while preserving edge categories and attributes."""
    forward_etypes = [et for et in list(data.edge_types) if not et[1].startswith("rev_")]
    for src_t, rel, dst_t in forward_etypes:
        rev_etype = (dst_t, f"rev_{rel}", src_t)
        if rev_etype not in CANONICAL_EDGE_TYPES:
            continue
        fstore = data[(src_t, rel, dst_t)]
        rstore = data[rev_etype]
        rstore.edge_category = CANONICAL_EDGE_TYPES[rev_etype]
        rstore.feature_names = list(fstore.feature_names)
        rstore.uses_target_expression = bool(getattr(fstore, "uses_target_expression", False))
        ei = fstore.edge_index
        rstore.edge_index = torch.stack([ei[1], ei[0]], dim=0).clone() if ei.numel() > 0 else ei.clone()
        rstore.edge_attr = fstore.edge_attr.clone()
        rstore.edge_attr_missing_mask = fstore.edge_attr_missing_mask.clone()
        rstore.evidence_tier_id = fstore.evidence_tier_id.clone()
        rstore.evidence_tier_labels = list(fstore.evidence_tier_labels)
        rstore.is_candidate_only = fstore.is_candidate_only.clone()


def from_utrnet_directory(
    gene_dir: Path | str,
    population: str | None = None,
    *,
    include_reverse_edges: bool = True,
    strip_expression_from_candidate_edges: bool = False,
) -> HeteroData:
    """Convert a UTRNet gene output directory of CSV tables into a `cell-regulatory-state/v0.1` `HeteroData` graph.

    Parameters
    ----------
    gene_dir : Path | str
        Directory containing `apa_profiles.csv`, `utr_isoforms.csv`, `regulatory_sites.csv`,
        `regulator_abundance.csv`, `interaction_scores.csv`, and `pas_sites.csv`.
    population : str | None
        Optional single curated population name (`celltype_corrected`). If `None`, builds a
        multi-population regulatory state graph across all populations in `apa_profiles.csv`.
    include_reverse_edges : bool
        If True, includes reverse relation edges (`rev_*`) so all node types receive messages.
    strip_expression_from_candidate_edges : bool
        If True, zeroes out `interaction_score` on candidate edges and marks it missing so only
        expression-free sequence/accessibility/evidence priors remain on candidate edges.
    """
    gene_dir = Path(gene_dir)
    profiles = pd.read_csv(gene_dir / "apa_profiles.csv")
    isoforms = pd.read_csv(gene_dir / "utr_isoforms.csv")
    reg_sites = pd.read_csv(gene_dir / "regulatory_sites.csv")
    abundance = pd.read_csv(gene_dir / "regulator_abundance.csv")
    score_cols = [
        "population",
        "isoform_id",
        "site_id",
        "regulatory_site_id",
        "regulator",
        "regulator_class",
        "model_weight",
        "accessibility",
        "evidence_factor",
        "evidence_tier",
        "abundance_factor",
        "interaction_score",
        "score_status",
    ]
    scores = pd.read_csv(gene_dir / "interaction_scores.csv", usecols=lambda c: c in score_cols)
    pas_sites = pd.read_csv(gene_dir / "pas_sites.csv") if (gene_dir / "pas_sites.csv").exists() else pd.DataFrame()
    cpa_edges = (
        pd.read_csv(gene_dir / "cpa_regulatory_edges.csv")
        if (gene_dir / "cpa_regulatory_edges.csv").exists()
        else pd.DataFrame()
    )
    cpa_expr = (
        pd.read_csv(gene_dir / "cpa_expression.csv")
        if (gene_dir / "cpa_expression.csv").exists()
        else pd.DataFrame()
    )

    if population is not None:
        profiles = profiles[profiles["celltype_corrected"] == population].copy()
        scores = scores[scores["population"] == population].copy()
        abundance = abundance[abundance["population"] == population].copy()
        if not cpa_expr.empty and "population" in cpa_expr.columns:
            cpa_expr = cpa_expr[cpa_expr["population"] == population].copy()
        if profiles.empty:
            raise ValueError(f"Unknown population {population!r} in {gene_dir}")

    # Fast vectorized motif survival probability and fractional cis-dosage lookup
    survival_lookup: dict[tuple[str, str], tuple[float, str]] = {}
    dosage_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    if not pas_sites.empty and not reg_sites.empty and not profiles.empty:
        pas_df = pas_sites[["site_id", "start", "strand"]].drop_duplicates("site_id").copy()
        pas_df["cleavage_coord"] = pas_df["start"].astype(float)
        strand = str(reg_sites["strand"].iloc[0])
        prox_coord = float(pas_df["cleavage_coord"].max() if strand == "-" else pas_df["cleavage_coord"].min())
        dist_coord = float(pas_df["cleavage_coord"].min() if strand == "-" else pas_df["cleavage_coord"].max())

        u_motifs = reg_sites.drop_duplicates("regulatory_site_id")[
            ["regulatory_site_id", "regulator", "genomic_start", "genomic_end"]
        ].copy()
        m_starts = u_motifs["genomic_start"].astype(float).to_numpy()
        m_ends = u_motifs["genomic_end"].astype(float).to_numpy()
        if strand == "-":
            r_types = np.where(m_starts >= prox_coord, "cUTR", np.where(m_ends <= dist_coord, "post_distal", "aUTR"))
        else:
            r_types = np.where(m_ends <= prox_coord, "cUTR", np.where(m_starts >= dist_coord, "post_distal", "aUTR"))
        valid_mask = r_types != "post_distal"
        u_motifs = u_motifs.loc[valid_mask].reset_index(drop=True)
        m_starts = m_starts[valid_mask]
        m_ends = m_ends[valid_mask]
        r_types = r_types[valid_mask]
        rsids_arr = u_motifs["regulatory_site_id"].astype(str).to_numpy()
        regs_arr = u_motifs["regulator"].astype(str).to_numpy()
        is_cutr = r_types == "cUTR"

        usage_merged = profiles[["celltype_corrected", "site_id", "pas_proportion"]].merge(
            pas_df[["site_id", "cleavage_coord"]], on="site_id", how="inner"
        )
        for pop, pop_grp in usage_merged.groupby("celltype_corrected"):
            coords = pop_grp["cleavage_coord"].to_numpy()
            thetas = pop_grp["pas_proportion"].fillna(0.0).to_numpy(dtype=float)
            tot_th = float(thetas.sum())
            if tot_th > 0:
                thetas = thetas / tot_th
            if strand == "-":
                retained = coords[None, :] <= m_starts[:, None]
            else:
                retained = coords[None, :] >= m_ends[:, None]
            surv_vec = np.where(is_cutr, 1.0, (retained * thetas[None, :]).sum(axis=1)).clip(0.0, 1.0)
            pop_str = str(pop)
            for idx_m in range(len(rsids_arr)):
                survival_lookup[(rsids_arr[idx_m], pop_str)] = (float(surv_vec[idx_m]), str(r_types[idx_m]))
            # Aggregate per regulator
            df_m = pd.DataFrame({"reg": regs_arr, "surv": surv_vec, "is_autr": ~is_cutr})
            for reg_k, rgrp in df_m.groupby("reg"):
                tot_d = float(rgrp["surv"].sum())
                autr_d = float(rgrp.loc[rgrp["is_autr"], "surv"].sum())
                dosage_lookup[(str(reg_k), pop_str)] = (tot_d, autr_d)

    data = HeteroData()
    data.schema_version = SCHEMA_VERSION
    data.model_name = "utrnnet-v0.1"
    data.is_temporal_observation = False
    data.source_directory = str(gene_dir)
    data.population_filter = population or "all"

    gene_id = (
        str(profiles["gene_id"].iloc[0]).split(".")[0]
        if not profiles.empty and "gene_id" in profiles.columns
        else str(isoforms["gene_id"].iloc[0]).split(".")[0]
    )
    data.gene_id = gene_id

    # 1. Population nodes
    populations = sorted(profiles["celltype_corrected"].unique())
    pop_ids: list[str] = []
    pop_feats: list[list[float]] = []
    pop_masks: list[list[bool]] = []
    for pop in populations:
        sub_p = profiles[profiles["celltype_corrected"] == pop]
        pop_ids.append(f"population:{pop}")
        n_cells_val, n_cells_miss = _log1p_safe(sub_p["n_cells"].iloc[0] if "n_cells" in sub_p.columns else None)
        meets_min = 1.0 if (not n_cells_miss and float(sub_p["n_cells"].iloc[0]) >= 15.0) else 0.0
        si_val, si_miss = _safe_float(sub_p["si"].iloc[0] if "si" in sub_p.columns else None)
        mag_val, mag_miss = _safe_float(
            sub_p["magnetization_genomic"].iloc[0] if "magnetization_genomic" in sub_p.columns else None
        )
        ent_val, ent_miss = _safe_float(sub_p["entropy"].iloc[0] if "entropy" in sub_p.columns else None)
        pop_feats.append([n_cells_val, meets_min, si_val, mag_val, ent_val])
        pop_masks.append([n_cells_miss, n_cells_miss, si_miss, mag_miss, ent_miss])
    pop_idx = _set_node_store(data, "population", pop_ids, pop_feats, pop_masks)

    # 2. Gene node
    gene_node_id = f"gene:{gene_id}"
    if "total_gene_umi" in profiles.columns and "n_cells" in profiles.columns:
        per_pop_mean_umi = profiles.groupby("celltype_corrected").apply(
            lambda g: float(g["total_gene_umi"].iloc[0]) / max(float(g["n_cells"].iloc[0]), 1.0),
            include_groups=False,
        )
        mean_gene_umi, gene_umi_miss = _log1p_safe(per_pop_mean_umi.mean())
    else:
        mean_gene_umi, gene_umi_miss = 0.0, True

    n_pas = float(profiles["site_id"].nunique()) if not profiles.empty else float(isoforms["site_id"].nunique())
    n_iso = float(isoforms["isoform_id"].nunique()) if not isoforms.empty else 0.0
    g_si, g_si_miss = _safe_float(profiles["si"].dropna().mean() if "si" in profiles.columns else None)
    g_mag, g_mag_miss = _safe_float(
        profiles["magnetization_genomic"].dropna().mean() if "magnetization_genomic" in profiles.columns else None
    )
    g_ent, g_ent_miss = _safe_float(profiles["entropy"].dropna().mean() if "entropy" in profiles.columns else None)
    gene_idx = _set_node_store(
        data,
        "gene",
        [gene_node_id],
        [[mean_gene_umi, math.log1p(n_pas), math.log1p(n_iso), g_si, g_mag, g_ent]],
        [[gene_umi_miss, False, False, g_si_miss, g_mag_miss, g_ent_miss]],
    )

    # 3. PAS nodes
    pas_coord_map: dict[str, tuple[float, float]] = {}
    if not pas_sites.empty and "start" in pas_sites.columns:
        starts = pas_sites["start"].astype(float)
        c_min, c_max = float(starts.min()), float(starts.max())
        span = max(c_max - c_min, 1.0)
        for r in pas_sites.itertuples(index=False):
            rel_c = (float(r.start) - c_min) / span
            s_sign = -1.0 if str(r.strand) == "-" else 1.0
            pas_coord_map[str(r.site_id)] = (rel_c, s_sign)

    unique_sites = sorted(profiles["site_id"].unique())
    pas_ids: list[str] = []
    pas_feats: list[list[float]] = []
    pas_masks: list[list[bool]] = []
    for sid in unique_sites:
        pas_ids.append(f"pas:{sid}")
        sub_s = profiles[profiles["site_id"] == sid]
        rel_c, s_sign = pas_coord_map.get(str(sid), (0.0, 0.0))
        coord_miss = str(sid) not in pas_coord_map
        prop_series = sub_s["pas_proportion"].dropna() if "pas_proportion" in sub_s.columns else pd.Series(dtype=float)
        if prop_series.empty:
            p_val, p_miss = 0.0, True
        else:
            p_val, p_miss = float(prop_series.mean()), False
        umi_val, umi_miss = _log1p_safe(sub_s["umi_sum"].mean() if "umi_sum" in sub_s.columns else None)
        if "usage_ci_high" in sub_s.columns and "usage_ci_low" in sub_s.columns:
            ci_w_series = (sub_s["usage_ci_high"] - sub_s["usage_ci_low"]).dropna()
            ci_w, ci_miss = (float(ci_w_series.mean()), False) if not ci_w_series.empty else (0.0, True)
        else:
            ci_w, ci_miss = 0.0, True
        pas_feats.append([rel_c, s_sign, p_val, umi_val, ci_w])
        pas_masks.append([coord_miss, coord_miss, p_miss, umi_miss, ci_miss])
    pas_idx = _set_node_store(data, "pas", pas_ids, pas_feats, pas_masks)

    # 4. UTR Isoform nodes
    iso_sub = isoforms[isoforms["site_id"].isin(unique_sites)].copy() if not isoforms.empty else pd.DataFrame()
    iso_ids: list[str] = []
    iso_feats: list[list[float]] = []
    iso_masks: list[list[bool]] = []
    ambiguity_codes: list[int] = []
    ambiguity_labels: list[str] = []
    span_codes: list[int] = []
    unresolved_flags: list[bool] = []

    pas_prop_lookup = (
        profiles.groupby("site_id")["pas_proportion"].mean().to_dict() if not profiles.empty else {}
    )
    for row in iso_sub.itertuples(index=False):
        iid = str(row.isoform_id)
        iso_ids.append(iid)
        seq_len = int(getattr(row, "sequence_length", 0))
        amb_str = str(getattr(row, "ambiguity_status", "unique"))
        span_str = str(getattr(row, "span_status", "within_span_limit"))
        amb_code = AMBIGUITY_STATUS_TO_CODE.get(amb_str, 1)
        span_code = SPAN_STATUS_TO_CODE.get(span_str, 0)
        is_unres = seq_len <= 0 or amb_code == 2

        mw, mw_miss = _safe_float(getattr(row, "model_weight", None))
        pp, pp_miss = _safe_float(pas_prop_lookup.get(row.site_id, None))
        iso_feats.append(
            [
                float(math.log1p(max(seq_len, 0))),
                mw,
                pp,
                1.0 if amb_code == 0 else 0.0,
                1.0 if amb_code == 1 else 0.0,
                1.0 if is_unres else 0.0,
                1.0 if span_code == 1 else 0.0,
            ]
        )
        iso_masks.append([is_unres, mw_miss, pp_miss, False, False, False, False])
        ambiguity_codes.append(amb_code)
        ambiguity_labels.append(amb_str)
        span_codes.append(span_code)
        unresolved_flags.append(is_unres)

    iso_idx = _set_node_store(
        data,
        "utr_isoform",
        iso_ids,
        iso_feats,
        iso_masks,
        extra={
            "ambiguity_code": torch.tensor(ambiguity_codes, dtype=torch.long),
            "ambiguity_labels": ambiguity_labels,
            "span_status_code": torch.tensor(span_codes, dtype=torch.long),
            "is_unresolved_mask": torch.tensor(unresolved_flags, dtype=torch.bool),
        },
    )

    # 5. Regulatory region nodes
    reg_Node_ids: list[str] = []
    reg_feats: list[list[float]] = []
    reg_masks: list[list[bool]] = []
    if not scores.empty:
        active_site_ids = sorted(scores["regulatory_site_id"].astype(str).unique())
    elif not reg_sites.empty and "regulatory_site_id" in reg_sites.columns:
        active_site_ids = sorted(reg_sites["regulatory_site_id"].astype(str).unique())
    else:
        active_site_ids = []

    site_meta = (
        reg_sites.drop_duplicates("regulatory_site_id").set_index("regulatory_site_id")
        if not reg_sites.empty and "regulatory_site_id" in reg_sites.columns
        else pd.DataFrame()
    )
    if not site_meta.empty and "genomic_start" in site_meta.columns:
        r_min = float(site_meta["genomic_start"].astype(float).min())
        r_max = float(site_meta["genomic_end"].astype(float).max())
        r_span = max(r_max - r_min, 1.0)
    else:
        r_min, r_span = 0.0, 1.0

    # Pre-extract fallback site attributes from scores once (avoiding O(N_sites * N_rows) filtering)
    score_site_fallback = (
        scores.drop_duplicates("regulatory_site_id").set_index("regulatory_site_id")
        if not scores.empty and "regulatory_site_id" in scores.columns
        else pd.DataFrame()
    )

    for rsid in active_site_ids:
        reg_Node_ids.append(f"region:{rsid}")
        if not site_meta.empty and rsid in site_meta.index:
            m_row = site_meta.loc[rsid]
            g_start, gs_miss = _safe_float(m_row.get("genomic_start", None))
            g_end, ge_miss = _safe_float(m_row.get("genomic_end", None))
            rel_pos = (g_start - r_min) / r_span if not gs_miss else 0.0
            length_bp = max(g_end - g_start + 1.0, 1.0) if not (gs_miss or ge_miss) else 0.0
            acc_val, acc_miss = _safe_float(m_row.get("accessibility", None))
            ev_tier = str(m_row.get("evidence_tier", "unknown"))
            ev_factor = EVIDENCE_TIER_FACTORS.get(ev_tier, 0.25)
        elif not score_site_fallback.empty and rsid in score_site_fallback.index:
            s_row = score_site_fallback.loc[rsid]
            rel_pos, gs_miss = 0.0, True
            length_bp, ge_miss = 0.0, True
            acc_val, acc_miss = _safe_float(s_row.get("accessibility", None))
            ev_tier = str(s_row.get("evidence_tier", "unknown"))
            ev_factor = EVIDENCE_TIER_FACTORS.get(ev_tier, 0.25)
        else:
            rel_pos, gs_miss = 0.0, True
            length_bp, ge_miss = 0.0, True
            acc_val, acc_miss = 0.0, True
            ev_factor = 0.25

        # Aggregate survival probability across selected populations
        surv_vals = [
            survival_lookup[(rsid, pop)][0]
            for pop in populations
            if (rsid, pop) in survival_lookup
        ]
        reg_types = [
            survival_lookup[(rsid, pop)][1]
            for pop in populations
            if (rsid, pop) in survival_lookup
        ]
        if surv_vals:
            surv_prob, surv_miss = float(np.mean(surv_vals)), False
            is_autr = 1.0 if any(rt == "aUTR" for rt in reg_types) else 0.0
        else:
            surv_prob, surv_miss = 0.0, True
            is_autr = 0.0

        reg_feats.append([rel_pos, float(math.log1p(length_bp)), acc_val, surv_prob, is_autr, ev_factor])
        reg_masks.append([gs_miss, gs_miss or ge_miss, acc_miss, surv_miss, surv_miss, False])

    region_idx = _set_node_store(data, "regulatory_region", reg_Node_ids, reg_feats, reg_masks)

    # 6. Regulator nodes (RBP & miRNA) — vectorized via groupby
    regulator_ids: list[str] = []
    regulator_feats: list[list[float]] = []
    regulator_masks: list[list[bool]] = []
    regulator_classes: list[str] = []

    ab_by_reg = (
        {str(k): grp for k, grp in abundance.groupby("regulator")}
        if not abundance.empty and "regulator" in abundance.columns
        else {}
    )
    sc_by_reg = (
        {str(k): grp for k, grp in scores.groupby("regulator", sort=True)}
        if not scores.empty and "regulator" in scores.columns
        else {}
    )
    for reg_name, sub_sc in sc_by_reg.items():
        regulator_ids.append(f"regulator:{reg_name}")
        reg_cls = str(sub_sc["regulator_class"].iloc[0]) if not sub_sc.empty else "RBP"
        regulator_classes.append(reg_cls)
        is_mirna = "mirna" in reg_cls.lower()
        is_rbp = not is_mirna

        sub_ab = ab_by_reg.get(reg_name, pd.DataFrame())
        if is_mirna or sub_ab.empty or sub_ab["mean_umi_per_cell"].dropna().empty:
            umi_log, umi_miss = 0.0, True
            ab_fac, ab_miss = 0.0, True
        else:
            umi_log, umi_miss = _log1p_safe(sub_ab["mean_umi_per_cell"].dropna().mean())
            ab_series = sub_sc["abundance_factor"].dropna()
            ab_fac, ab_miss = (float(ab_series.mean()), False) if not ab_series.empty else (0.0, True)

        max_ev = float(sub_sc["evidence_factor"].max()) if not sub_sc.empty else 0.25
        dos_totals = [dosage_lookup[(reg_name, p)][0] for p in populations if (reg_name, p) in dosage_lookup]
        dos_autrs = [dosage_lookup[(reg_name, p)][1] for p in populations if (reg_name, p) in dosage_lookup]
        if dos_totals:
            d_tot, d_autr, d_miss = float(np.mean(dos_totals)), float(np.mean(dos_autrs)), False
        else:
            d_tot, d_autr, d_miss = 0.0, 0.0, True

        regulator_feats.append(
            [umi_log, ab_fac, 1.0 if is_rbp else 0.0, 1.0 if is_mirna else 0.0, max_ev, d_tot, d_autr]
        )
        regulator_masks.append([umi_miss, ab_miss, False, False, False, d_miss, d_miss])

    regulator_idx = _set_node_store(
        data,
        "regulator",
        regulator_ids,
        regulator_feats,
        regulator_masks,
        extra={"regulator_classes": regulator_classes},
    )

    # 7. Optional CPA factor nodes
    cpa_ids: list[str] = []
    cpa_feats: list[list[float]] = []
    cpa_masks: list[list[bool]] = []
    if not cpa_edges.empty:
        for row in cpa_edges.drop_duplicates("source").itertuples(index=False):
            cid = str(row.source)
            fac = str(row.factor)
            cpa_ids.append(cid)
            sub_ce = cpa_expr[cpa_expr["factor"] == fac] if not cpa_expr.empty else pd.DataFrame()
            if not sub_ce.empty:
                m_exp, m_miss = _log1p_safe(sub_ce["mean_expr"].dropna().mean())
                pct_e, p_miss = _safe_float(sub_ce["pct_expressing"].dropna().mean())
            else:
                m_exp, m_miss = 0.0, True
                pct_e, p_miss = 0.0, True
            cpa_feats.append([m_exp, pct_e])
            cpa_masks.append([m_miss, p_miss])
    cpa_idx = _set_node_store(data, "cpa_factor", cpa_ids, cpa_feats, cpa_masks)

    # --- Build Typed Edges ---

    # E1: ("gene", "has_pas", "pas") [structural]
    e_g_p_src, e_g_p_dst, e_g_p_f, e_g_p_m, e_g_p_t = [], [], [], [], []
    for sid in unique_sites:
        e_g_p_src.append(gene_idx[gene_node_id])
        e_g_p_dst.append(pas_idx[f"pas:{sid}"])
        e_g_p_f.append([1.0, 1.0])
        e_g_p_m.append([False, False])
        e_g_p_t.append("observed_reference")
    _set_edge_store(data, ("gene", "has_pas", "pas"), e_g_p_src, e_g_p_dst, e_g_p_f, e_g_p_m, e_g_p_t)

    # E2: ("pas", "defines_isoform", "utr_isoform") [annotation_derived]
    e_p_i_src, e_p_i_dst, e_p_i_f, e_p_i_m, e_p_i_t = [], [], [], [], []
    for row in iso_sub.itertuples(index=False):
        pas_key = f"pas:{row.site_id}"
        if pas_key not in pas_idx or str(row.isoform_id) not in iso_idx:
            continue
        mw, mw_miss = _safe_float(getattr(row, "model_weight", None))
        amb_code = AMBIGUITY_STATUS_TO_CODE.get(str(getattr(row, "ambiguity_status", "unique")), 1)
        is_unres = int(getattr(row, "sequence_length", 0)) <= 0 or amb_code == 2
        e_p_i_src.append(pas_idx[pas_key])
        e_p_i_dst.append(iso_idx[str(row.isoform_id)])
        e_p_i_f.append([mw, 1.0 if amb_code == 1 else 0.0, 1.0 if is_unres else 0.0, 1.0])
        e_p_i_m.append([mw_miss, False, False, False])
        e_p_i_t.append("annotation_model")
    _set_edge_store(data, ("pas", "defines_isoform", "utr_isoform"), e_p_i_src, e_p_i_dst, e_p_i_f, e_p_i_m, e_p_i_t)

    # E3: ("utr_isoform", "contains_site", "regulatory_region") [structural]
    e_i_r_src, e_i_r_dst, e_i_r_f, e_i_r_m, e_i_r_t = [], [], [], [], []
    seen_i_r: set[tuple[int, int]] = set()
    site_source_df = scores if not scores.empty else reg_sites
    if not site_source_df.empty and "isoform_id" in site_source_df.columns and "regulatory_site_id" in site_source_df.columns:
        for row in site_source_df[["isoform_id", "regulatory_site_id"]].drop_duplicates().itertuples(index=False):
            iid = str(row.isoform_id)
            rk = f"region:{row.regulatory_site_id}"
            if iid in iso_idx and rk in region_idx:
                pair = (iso_idx[iid], region_idx[rk])
                if pair not in seen_i_r:
                    seen_i_r.add(pair)
                    e_i_r_src.append(pair[0])
                    e_i_r_dst.append(pair[1])
                    e_i_r_f.append([1.0, 1.0])
                    e_i_r_m.append([False, False])
                    e_i_r_t.append("sequence_or_import")
    _set_edge_store(
        data, ("utr_isoform", "contains_site", "regulatory_region"), e_i_r_src, e_i_r_dst, e_i_r_f, e_i_r_m, e_i_r_t
    )

    # E4: ("population", "uses_isoform", "utr_isoform") [observed]
    e_pop_i_src, e_pop_i_dst, e_pop_i_f, e_pop_i_m, e_pop_i_t = [], [], [], [], []
    for row in profiles.itertuples(index=False):
        pk = f"population:{row.celltype_corrected}"
        if pk not in pop_idx:
            continue
        pp, pp_miss = _safe_float(getattr(row, "pas_proportion", None))
        umi_l, umi_miss = _log1p_safe(getattr(row, "umi_sum", None))
        ci_h, ch_miss = _safe_float(getattr(row, "usage_ci_high", None))
        ci_l, cl_miss = _safe_float(getattr(row, "usage_ci_low", None))
        ci_w = max(ci_h - ci_l, 0.0) if not (ch_miss or cl_miss) else 0.0
        for iso_row in iso_sub[iso_sub["site_id"] == row.site_id].itertuples(index=False):
            iid = str(iso_row.isoform_id)
            if iid not in iso_idx:
                continue
            e_pop_i_src.append(pop_idx[pk])
            e_pop_i_dst.append(iso_idx[iid])
            e_pop_i_f.append([pp, umi_l, ci_w, 1.0])
            e_pop_i_m.append([pp_miss, umi_miss, ch_miss or cl_miss, False])
            e_pop_i_t.append("Sierra_UMI")
    _set_edge_store(
        data, ("population", "uses_isoform", "utr_isoform"), e_pop_i_src, e_pop_i_dst, e_pop_i_f, e_pop_i_m, e_pop_i_t
    )

    # E5: ("population", "expresses_gene", "gene") [observed]
    e_pop_g_src, e_pop_g_dst, e_pop_g_f, e_pop_g_m, e_pop_g_t = [], [], [], [], []
    for pop in populations:
        sub_p = profiles[profiles["celltype_corrected"] == pop]
        pk = f"population:{pop}"
        tot_umi, tu_miss = _safe_float(sub_p["total_gene_umi"].iloc[0] if "total_gene_umi" in sub_p.columns else None)
        n_c, nc_miss = _safe_float(sub_p["n_cells"].iloc[0] if "n_cells" in sub_p.columns else None)
        mean_u = tot_umi / max(n_c, 1.0) if not (tu_miss or nc_miss) else 0.0
        log_tot, lt_miss = _log1p_safe(tot_umi if not tu_miss else None)
        e_pop_g_src.append(pop_idx[pk])
        e_pop_g_dst.append(gene_idx[gene_node_id])
        e_pop_g_f.append([float(math.log1p(max(mean_u, 0.0))), log_tot, 0.0, 1.0])
        e_pop_g_m.append([tu_miss or nc_miss, lt_miss, True, False])
        e_pop_g_t.append("curated_UMI")
    _set_edge_store(
        data,
        ("population", "expresses_gene", "gene"),
        e_pop_g_src,
        e_pop_g_dst,
        e_pop_g_f,
        e_pop_g_m,
        e_pop_g_t,
        uses_target_expression=True,
    )

    # E6: ("population", "expresses_regulator", "regulator") [observed]
    e_pop_r_src, e_pop_r_dst, e_pop_r_f, e_pop_r_m, e_pop_r_t = [], [], [], [], []
    if not abundance.empty:
        for row in abundance.itertuples(index=False):
            pk = f"population:{row.population}"
            rk = f"regulator:{row.regulator}"
            if pk not in pop_idx or rk not in regulator_idx:
                continue
            m_umi, mu_miss = _safe_float(getattr(row, "mean_umi_per_cell", None))
            l_umi, lu_miss = _log1p_safe(getattr(row, "mean_umi_per_cell", None))
            e_pop_r_src.append(pop_idx[pk])
            e_pop_r_dst.append(regulator_idx[rk])
            e_pop_r_f.append([m_umi, l_umi, 0.0, 1.0])
            e_pop_r_m.append([mu_miss, lu_miss, True, False])
            e_pop_r_t.append("curated_UMI")
    _set_edge_store(
        data,
        ("population", "expresses_regulator", "regulator"),
        e_pop_r_src,
        e_pop_r_dst,
        e_pop_r_f,
        e_pop_r_m,
        e_pop_r_t,
    )

    # E7: ("regulatory_region", "candidate_binding", "regulator") [inferred_candidate] — vectorized
    e_r_reg_src, e_r_reg_dst, e_r_reg_f, e_r_reg_m, e_r_reg_t = [], [], [], [], []
    if not scores.empty:
        scores_tmp = scores.copy()
        scores_tmp["_is_comp"] = (scores_tmp["score_status"] == "complete_prediction").astype(float)
        agg_bind = scores_tmp.groupby(["regulatory_site_id", "regulator"], sort=True, as_index=False).agg(
            evidence_tier=("evidence_tier", "first"),
            evidence_factor=("evidence_factor", "first"),
            model_weight=("model_weight", "first"),
            accessibility=("accessibility", "mean"),
            interaction_score=("interaction_score", "mean"),
            is_comp=("_is_comp", "min"),
        )
        for row in agg_bind.itertuples(index=False):
            rk = f"region:{row.regulatory_site_id}"
            regk = f"regulator:{row.regulator}"
            if rk not in region_idx or regk not in regulator_idx:
                continue
            ev_tier = str(row.evidence_tier)
            ev_fac = float(row.evidence_factor)
            mw = float(row.model_weight) if pd.notna(row.model_weight) else 1.0
            acc_val, acc_miss = _safe_float(row.accessibility)
            seq_prior = mw * (acc_val if not acc_miss else 1.0) * ev_fac
            if strip_expression_from_candidate_edges:
                iscore, iscore_miss = 0.0, True
            else:
                iscore, iscore_miss = _safe_float(row.interaction_score)
            is_comp = float(row.is_comp)
            e_r_reg_src.append(region_idx[rk])
            e_r_reg_dst.append(regulator_idx[regk])
            e_r_reg_f.append([seq_prior, acc_val, ev_fac, iscore, is_comp])
            e_r_reg_m.append([False, acc_miss, False, iscore_miss, False])
            e_r_reg_t.append(ev_tier)
    _set_edge_store(
        data,
        ("regulatory_region", "candidate_binding", "regulator"),
        e_r_reg_src,
        e_r_reg_dst,
        e_r_reg_f,
        e_r_reg_m,
        e_r_reg_t,
        uses_target_expression=False,
    )

    # E8: ("regulator", "candidate_regulation", "gene") [inferred_candidate] — vectorized
    e_reg_g_src, e_reg_g_dst, e_reg_g_f, e_reg_g_m, e_reg_g_t = [], [], [], [], []
    if not scores.empty:
        pop_reg_sums = (
            scores.groupby(["regulator", "population"], as_index=False)["interaction_score"]
            .sum()
            .groupby("regulator")["interaction_score"]
            .mean()
            .to_dict()
        )
        for reg_name, grp in sc_by_reg.items():
            regk = f"regulator:{reg_name}"
            if regk not in regulator_idx:
                continue
            ev_tier = str(grp["evidence_tier"].iloc[0])
            ev_fac = float(grp["evidence_factor"].max())
            acc_val, acc_miss = _safe_float(grp["accessibility"].dropna().mean() if "accessibility" in grp.columns else None)
            dos_vals = [dosage_lookup[(str(reg_name), p)][0] for p in populations if (str(reg_name), p) in dosage_lookup]
            cis_prior = float(np.mean(dos_vals)) if dos_vals else float(grp["regulatory_site_id"].nunique())
            if strip_expression_from_candidate_edges:
                iscore, iscore_miss = 0.0, True
            else:
                iscore, iscore_miss = _safe_float(pop_reg_sums.get(reg_name, None))
            is_comp = 1.0 if (grp["score_status"] == "complete_prediction").all() else 0.0
            e_reg_g_src.append(regulator_idx[regk])
            e_reg_g_dst.append(gene_idx[gene_node_id])
            e_reg_g_f.append([cis_prior, acc_val, ev_fac, iscore, is_comp])
            e_reg_g_m.append([False, acc_miss, False, iscore_miss, False])
            e_reg_g_t.append(ev_tier)
    _set_edge_store(
        data,
        ("regulator", "candidate_regulation", "gene"),
        e_reg_g_src,
        e_reg_g_dst,
        e_reg_g_f,
        e_reg_g_m,
        e_reg_g_t,
        uses_target_expression=False,
    )

    # E9: ("cpa_factor", "modulates_apa", "gene") [inferred_candidate]
    e_cpa_g_src, e_cpa_g_dst, e_cpa_g_f, e_cpa_g_m, e_cpa_g_t = [], [], [], [], []
    if not cpa_edges.empty:
        for row in cpa_edges.itertuples(index=False):
            cid = str(row.source)
            if cid not in cpa_idx:
                continue
            corr_r, r_miss = _safe_float(getattr(row, "correlation_r", None))
            pval, p_miss = _safe_float(getattr(row, "p_value", None))
            ev_tier = str(getattr(row, "evidence_tier", "experimental_other_context"))
            ev_fac = EVIDENCE_TIER_FACTORS.get(ev_tier, 0.75)
            e_cpa_g_src.append(cpa_idx[cid])
            e_cpa_g_dst.append(gene_idx[gene_node_id])
            e_cpa_g_f.append([ abs(corr_r), 1.0 - pval if not p_miss else 0.0, ev_fac, corr_r, 1.0 if not r_miss else 0.0])
            e_cpa_g_m.append([r_miss, p_miss, False, r_miss, False])
            e_cpa_g_t.append(ev_tier)
    _set_edge_store(
        data,
        ("cpa_factor", "modulates_apa", "gene"),
        e_cpa_g_src,
        e_cpa_g_dst,
        e_cpa_g_f,
        e_cpa_g_m,
        e_cpa_g_t,
        uses_target_expression=False,
    )

    if include_reverse_edges:
        _add_reverse_edges(data)

    validate_cell_regulatory_state(data, raise_on_error=True)
    return data


def from_graphml(
    graph_or_path: Path | str | nx.Graph,
    *,
    include_reverse_edges: bool = True,
) -> HeteroData:
    """Convert a UTRNet GraphML file (or `nx.MultiDiGraph`) into a `cell-regulatory-state/v0.1` `HeteroData` graph."""
    if isinstance(graph_or_path, (str, Path)):
        nx_graph = nx.read_graphml(Path(graph_or_path))
        source_path = str(graph_or_path)
    else:
        nx_graph = graph_or_path
        source_path = "networkx_in_memory"

    data = HeteroData()
    data.schema_version = SCHEMA_VERSION
    data.model_name = str(nx_graph.graph.get("model", "utrnnet-v0.1"))
    data.is_temporal_observation = False
    data.source_directory = source_path
    data.population_filter = str(nx_graph.graph.get("population", "unknown"))

    # Group nodes by canonical node type
    grouped_nodes: dict[str, list[tuple[str, dict[str, Any]]]] = {nt: [] for nt in NODE_TYPES}
    for nid, attrs in sorted(nx_graph.nodes(data=True), key=lambda x: str(x[0])):
        kind = str(attrs.get("kind", ""))
        if kind == "population" or str(nid).startswith("population:"):
            grouped_nodes["population"].append((str(nid), attrs))
        elif kind == "gene" or str(nid).startswith("gene:"):
            grouped_nodes["gene"].append((str(nid), attrs))
        elif kind == "PAS" or str(nid).startswith("pas:"):
            grouped_nodes["pas"].append((str(nid), attrs))
        elif kind == "UTR_isoform" or str(nid).startswith("utr:"):
            grouped_nodes["utr_isoform"].append((str(nid), attrs))
        elif kind == "regulatory_region" or str(nid).startswith("region:"):
            grouped_nodes["regulatory_region"].append((str(nid), attrs))
        elif kind == "cpa_factor" or str(nid).startswith("cpa:"):
            grouped_nodes["cpa_factor"].append((str(nid), attrs))
        else:
            grouped_nodes["regulator"].append((str(nid), attrs))

    # Extract edge weights for node summary features when converting purely from GraphML
    iso_usage_from_edges: dict[str, tuple[float, float]] = {}
    for u, v, eattrs in nx_graph.edges(data=True):
        if eattrs.get("kind") == "uses_isoform":
            w, _ = _safe_float(eattrs.get("weight", None))
            umi, _ = _safe_float(eattrs.get("umi", None))
            iso_usage_from_edges[str(v)] = (w, umi)

    # 1. Population
    pop_ids, pop_f, pop_m = [], [], []
    for nid, attrs in grouped_nodes["population"]:
        pop_ids.append(nid)
        si, si_m = _safe_float(attrs.get("shortening_index", None))
        mag, mag_m = _safe_float(attrs.get("magnetization_genomic", None))
        ent, ent_m = _safe_float(attrs.get("entropy", None))
        pop_f.append([0.0, 1.0, si, mag, ent])
        pop_m.append([True, True, si_m, mag_m, ent_m])
    pop_idx = _set_node_store(data, "population", pop_ids, pop_f, pop_m)

    # 2. Gene
    gene_ids, gene_f, gene_m = [], [], []
    n_pas_total = float(len(grouped_nodes["pas"]))
    n_iso_total = float(len(grouped_nodes["utr_isoform"]))
    for nid, attrs in grouped_nodes["gene"]:
        gene_ids.append(nid)
        gene_f.append([0.0, math.log1p(n_pas_total), math.log1p(n_iso_total), 0.0, 0.0, 0.0])
        gene_m.append([True, False, False, True, True, True])
    gene_idx = _set_node_store(data, "gene", gene_ids, gene_f, gene_m)

    # 3. PAS
    pas_ids, pas_f, pas_m = [], [], []
    for nid, attrs in grouped_nodes["pas"]:
        pas_ids.append(nid)
        pas_f.append([0.0, 0.0, 0.0, 0.0, 0.0])
        pas_m.append([True, True, True, True, True])
    pas_idx = _set_node_store(data, "pas", pas_ids, pas_f, pas_m)

    # 4. UTR Isoform
    iso_ids, iso_f, iso_m = [], [], []
    amb_codes, amb_labels, span_codes, unres_flags = [], [], [], []
    for nid, attrs in grouped_nodes["utr_isoform"]:
        iso_ids.append(nid)
        seq_len_f, seq_m = _safe_float(attrs.get("sequence_length", 0))
        seq_len = int(seq_len_f)
        amb_str = str(attrs.get("ambiguity", "unique"))
        span_str = str(attrs.get("span_status", "within_span_limit"))
        amb_code = AMBIGUITY_STATUS_TO_CODE.get(amb_str, 1)
        span_code = SPAN_STATUS_TO_CODE.get(span_str, 0)
        is_unres = seq_len <= 0 or amb_code == 2 or seq_m
        pp, _ = iso_usage_from_edges.get(nid, (0.0, 0.0))
        pp_miss = nid not in iso_usage_from_edges
        iso_f.append(
            [
                float(math.log1p(max(seq_len, 0))),
                1.0,
                pp,
                1.0 if amb_code == 0 else 0.0,
                1.0 if amb_code == 1 else 0.0,
                1.0 if is_unres else 0.0,
                1.0 if span_code == 1 else 0.0,
            ]
        )
        iso_m.append([is_unres, True, pp_miss, False, False, False, False])
        amb_codes.append(amb_code)
        amb_labels.append(amb_str)
        span_codes.append(span_code)
        unres_flags.append(is_unres)
    iso_idx = _set_node_store(
        data,
        "utr_isoform",
        iso_ids,
        iso_f,
        iso_m,
        extra={
            "ambiguity_code": torch.tensor(amb_codes, dtype=torch.long),
            "ambiguity_labels": amb_labels,
            "span_status_code": torch.tensor(span_codes, dtype=torch.long),
            "is_unresolved_mask": torch.tensor(unres_flags, dtype=torch.bool),
        },
    )

    # 5. Regulatory region
    reg_ids, reg_f, reg_m = [], [], []
    for nid, attrs in grouped_nodes["regulatory_region"]:
        reg_ids.append(nid)
        reg_f.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.25])
        reg_m.append([True, True, True, True, True, False])
    region_idx = _set_node_store(data, "regulatory_region", reg_ids, reg_f, reg_m)

    # 6. Regulator
    r_ids, r_f, r_m, r_classes = [], [], [], []
    for nid, attrs in grouped_nodes["regulator"]:
        r_ids.append(nid)
        r_cls = str(attrs.get("kind", "RBP"))
        r_classes.append(r_cls)
        is_mirna = "mirna" in r_cls.lower() or "mir-" in nid.lower()
        is_rbp = not is_mirna
        r_f.append([0.0, 0.0, 1.0 if is_rbp else 0.0, 1.0 if is_mirna else 0.0, 0.25, 0.0, 0.0])
        r_m.append([True, True, False, False, False, True, True])
    regulator_idx = _set_node_store(
        data, "regulator", r_ids, r_f, r_m, extra={"regulator_classes": r_classes}
    )

    # 7. CPA factor
    c_ids, c_f, c_m = [], [], []
    for nid, attrs in grouped_nodes["cpa_factor"]:
        c_ids.append(nid)
        me, me_m = _log1p_safe(attrs.get("mean_expr", None))
        pe, pe_m = _safe_float(attrs.get("pct_expressing", None))
        c_f.append([me, pe])
        c_m.append([me_m, pe_m])
    cpa_idx = _set_node_store(data, "cpa_factor", c_ids, c_f, c_m)

    # Collect edges by canonical type
    edge_buckets: dict[tuple[str, str, str], tuple[list[int], list[int], list[list[float]], list[list[bool]], list[str]]] = {
        et: ([], [], [], [], []) for et in CANONICAL_EDGE_TYPES if not et[1].startswith("rev_")
    }

    for u, v, eattrs in nx_graph.edges(data=True):
        u_str, v_str = str(u), str(v)
        ekind = str(eattrs.get("kind", ""))
        ev_tier = str(eattrs.get("evidence", "unknown"))
        ev_fac = EVIDENCE_TIER_FACTORS.get(ev_tier, 0.25)

        if ekind == "gene_has_PAS" and u_str in gene_idx and v_str in pas_idx:
            b = edge_buckets[("gene", "has_pas", "pas")]
            b[0].append(gene_idx[u_str])
            b[1].append(pas_idx[v_str])
            b[2].append([1.0, 1.0])
            b[3].append([False, False])
            b[4].append(ev_tier)
        elif ekind == "PAS_defines_candidate_isoform" and u_str in pas_idx and v_str in iso_idx:
            b = edge_buckets[("pas", "defines_isoform", "utr_isoform")]
            dst_i = iso_idx[v_str]
            is_amb = 1.0 if int(data["utr_isoform"].ambiguity_code[dst_i]) == 1 else 0.0
            is_unr = 1.0 if bool(data["utr_isoform"].is_unresolved_mask[dst_i]) else 0.0
            b[0].append(pas_idx[u_str])
            b[1].append(dst_i)
            b[2].append([1.0, is_amb, is_unr, 1.0])
            b[3].append([True, False, False, False])
            b[4].append(ev_tier)
        elif ekind == "contains_site" and u_str in iso_idx and v_str in region_idx:
            b = edge_buckets[("utr_isoform", "contains_site", "regulatory_region")]
            b[0].append(iso_idx[u_str])
            b[1].append(region_idx[v_str])
            b[2].append([1.0, 1.0])
            b[3].append([False, False])
            b[4].append(ev_tier)
        elif ekind == "uses_isoform" and u_str in pop_idx and v_str in iso_idx:
            b = edge_buckets[("population", "uses_isoform", "utr_isoform")]
            w, w_m = _safe_float(eattrs.get("weight", None))
            umi_l, u_m = _log1p_safe(eattrs.get("umi", None))
            b[0].append(pop_idx[u_str])
            b[1].append(iso_idx[v_str])
            b[2].append([w, umi_l, 0.0, 1.0])
            b[3].append([w_m, u_m, True, False])
            b[4].append(ev_tier)
        elif ekind == "candidate_binding" and u_str in region_idx and v_str in regulator_idx:
            b = edge_buckets[("regulatory_region", "candidate_binding", "regulator")]
            w, w_m = _safe_float(eattrs.get("weight", None))
            s_stat = str(eattrs.get("score_status", ""))
            is_comp = 1.0 if s_stat == "complete_prediction" else 0.0
            b[0].append(region_idx[u_str])
            b[1].append(regulator_idx[v_str])
            b[2].append([ev_fac, 0.0, ev_fac, w, is_comp])
            b[3].append([False, True, False, w_m, False])
            b[4].append(ev_tier)
        elif ekind == "candidate_regulation" and u_str in regulator_idx and v_str in gene_idx:
            b = edge_buckets[("regulator", "candidate_regulation", "gene")]
            w, w_m = _safe_float(eattrs.get("weight", None))
            s_stat = str(eattrs.get("score_status", ""))
            is_comp = 1.0 if s_stat == "complete_prediction" else 0.0
            b[0].append(regulator_idx[u_str])
            b[1].append(gene_idx[v_str])
            b[2].append([ev_fac, 0.0, ev_fac, w, is_comp])
            b[3].append([False, True, False, w_m, False])
            b[4].append(ev_tier)
        elif u_str in cpa_idx and v_str in gene_idx:
            b = edge_buckets[("cpa_factor", "modulates_apa", "gene")]
            w, w_m = _safe_float(eattrs.get("weight", None))
            pv, pv_m = _safe_float(eattrs.get("p_value", None))
            b[0].append(cpa_idx[u_str])
            b[1].append(gene_idx[v_str])
            b[2].append([abs(w), 1.0 - pv if not pv_m else 0.0, ev_fac, w, 1.0 if not w_m else 0.0])
            b[3].append([w_m, pv_m, False, w_m, False])
            b[4].append(ev_tier)

    for etype, (src_l, dst_l, f_l, m_l, t_l) in edge_buckets.items():
        _set_edge_store(
            data,
            etype,
            src_l,
            dst_l,
            f_l,
            m_l,
            t_l,
            uses_target_expression=(etype == ("population", "expresses_gene", "gene")),
        )

    if include_reverse_edges:
        _add_reverse_edges(data)

    validate_cell_regulatory_state(data, raise_on_error=True)
    return data


def merge_hetero_graphs(graphs: list[HeteroData], *, include_reverse_edges: bool = True) -> HeteroData:
    """Merge multiple gene-level `HeteroData` graphs into a unified multi-gene `HeteroData` graph.

    Shared node types (`population`, `regulator`, `cpa_factor`) are deduplicated by stable `node_id`
    with observed features averaged over non-missing entries.
    """
    if not graphs:
        raise ValueError("Cannot merge an empty list of HeteroData graphs")
    if len(graphs) == 1:
        return graphs[0].clone()

    merged = HeteroData()
    merged.schema_version = SCHEMA_VERSION
    merged.model_name = "utrnnet-v0.1-multigene"
    merged.is_temporal_observation = any(bool(getattr(g, "is_temporal_observation", False)) for g in graphs)
    merged.source_directory = "merged_cohort"
    merged.population_filter = "merged"

    global_id_maps: dict[str, dict[str, int]] = {}
    for ntype in NODE_TYPES:
        feat_sum: dict[str, np.ndarray] = {}
        feat_count: dict[str, np.ndarray] = {}
        extra_amb: dict[str, int] = {}
        extra_amb_lbl: dict[str, str] = {}
        extra_span: dict[str, int] = {}
        extra_unres: dict[str, bool] = {}
        extra_reg_cls: dict[str, str] = {}
        ordered_ids: list[str] = []

        n_feats = len(NODE_FEATURE_SCHEMAS[ntype])
        for g in graphs:
            if ntype not in g.node_types:
                continue
            store = g[ntype]
            x_np = store.x.detach().cpu().numpy()
            m_np = store.x_missing_mask.detach().cpu().numpy()
            for idx, nid in enumerate(store.node_ids):
                if nid not in feat_sum:
                    ordered_ids.append(nid)
                    feat_sum[nid] = np.zeros(n_feats, dtype=np.float64)
                    feat_count[nid] = np.zeros(n_feats, dtype=np.int64)
                obs = ~m_np[idx]
                feat_sum[nid][obs] += x_np[idx][obs]
                feat_count[nid][obs] += 1
                if ntype == "utr_isoform":
                    extra_amb[nid] = int(store.ambiguity_code[idx])
                    extra_amb_lbl[nid] = str(store.ambiguity_labels[idx])
                    extra_span[nid] = int(store.span_status_code[idx])
                    extra_unres[nid] = bool(store.is_unresolved_mask[idx])
                elif ntype == "regulator" and hasattr(store, "regulator_classes"):
                    extra_reg_cls[nid] = str(store.regulator_classes[idx])

        feats_out: list[list[float]] = []
        masks_out: list[list[bool]] = []
        for nid in ordered_ids:
            cnt = feat_count[nid]
            miss = cnt == 0
            vals = np.where(miss, 0.0, feat_sum[nid] / np.maximum(cnt, 1))
            feats_out.append(vals.astype(float).tolist())
            masks_out.append(miss.astype(bool).tolist())

        extra_dict: dict[str, Any] = {}
        if ntype == "utr_isoform":
            extra_dict = {
                "ambiguity_code": torch.tensor([extra_amb[n] for n in ordered_ids], dtype=torch.long),
                "ambiguity_labels": [extra_amb_lbl[n] for n in ordered_ids],
                "span_status_code": torch.tensor([extra_span[n] for n in ordered_ids], dtype=torch.long),
                "is_unresolved_mask": torch.tensor([extra_unres[n] for n in ordered_ids], dtype=torch.bool),
            }
        elif ntype == "regulator":
            extra_dict = {"regulator_classes": [extra_reg_cls.get(n, "RBP") for n in ordered_ids]}

        global_id_maps[ntype] = _set_node_store(merged, ntype, ordered_ids, feats_out, masks_out, extra=extra_dict)

    # Merge forward edges and deduplicate by (src_idx, dst_idx)
    forward_etypes = [et for et in CANONICAL_EDGE_TYPES if not et[1].startswith("rev_")]
    for etype in forward_etypes:
        src_t, _, dst_t = etype
        seen_pairs: dict[tuple[int, int], int] = {}
        src_list: list[int] = []
        dst_list: list[int] = []
        ef_list: list[list[float]] = []
        em_list: list[list[bool]] = []
        et_list: list[str] = []
        uses_target_expr = False

        for g in graphs:
            if etype not in g.edge_types:
                continue
            estore = g[etype]
            uses_target_expr = uses_target_expr or bool(getattr(estore, "uses_target_expression", False))
            ei = estore.edge_index.detach().cpu().numpy()
            ea = estore.edge_attr.detach().cpu().numpy()
            em = estore.edge_attr_missing_mask.detach().cpu().numpy()
            src_ids_local = g[src_t].node_ids
            dst_ids_local = g[dst_t].node_ids
            t_labels = getattr(estore, "evidence_tier_labels", ["unknown"] * ei.shape[1])

            for k in range(ei.shape[1]):
                gs = global_id_maps[src_t][src_ids_local[int(ei[0, k])]]
                gd = global_id_maps[dst_t][dst_ids_local[int(ei[1, k])]]
                pair = (gs, gd)
                if pair not in seen_pairs:
                    seen_pairs[pair] = len(src_list)
                    src_list.append(gs)
                    dst_list.append(gd)
                    ef_list.append(ea[k].astype(float).tolist())
                    em_list.append(em[k].astype(bool).tolist())
                    et_list.append(str(t_labels[k]))

        _set_edge_store(
            merged,
            etype,
            src_list,
            dst_list,
            ef_list,
            em_list,
            et_list,
            uses_target_expression=uses_target_expr,
        )

    if include_reverse_edges:
        _add_reverse_edges(merged)

    validate_cell_regulatory_state(merged, raise_on_error=True)
    return merged
