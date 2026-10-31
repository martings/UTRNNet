"""Schema and validation contracts for `cell-regulatory-state/v0.1`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import torch
from torch_geometric.data import HeteroData

SCHEMA_VERSION = "cell-regulatory-state/v0.1"

NODE_TYPES: tuple[str, ...] = (
    "population",
    "gene",
    "pas",
    "utr_isoform",
    "regulatory_region",
    "regulator",
    "cpa_factor",
)

EDGE_CATEGORIES: tuple[str, ...] = (
    "structural",
    "annotation_derived",
    "observed",
    "inferred_candidate",
)

# Canonical relation schema: (src_type, rel_name, dst_type) -> edge_category
CANONICAL_EDGE_TYPES: dict[tuple[str, str, str], str] = {
    # 1. Structural edges (genomic / sequence containment)
    ("gene", "has_pas", "pas"): "structural",
    ("pas", "rev_has_pas", "gene"): "structural",
    ("utr_isoform", "contains_site", "regulatory_region"): "structural",
    ("regulatory_region", "rev_contains_site", "utr_isoform"): "structural",
    # 2. Annotation-derived edges (GTF/FASTA candidate 3'UTR models, with explicit ambiguity)
    ("pas", "defines_isoform", "utr_isoform"): "annotation_derived",
    ("utr_isoform", "rev_defines_isoform", "pas"): "annotation_derived",
    # 3. Observed edges (empirical UMI measurements from scPolASeq / Sierra / 10x)
    ("population", "uses_isoform", "utr_isoform"): "observed",
    ("utr_isoform", "rev_uses_isoform", "population"): "observed",
    ("population", "expresses_gene", "gene"): "observed",
    ("gene", "rev_expresses_gene", "population"): "observed",
    ("population", "expresses_regulator", "regulator"): "observed",
    ("regulator", "rev_expresses_regulator", "population"): "observed",
    # 4. Inferred / candidate edges (sequence motifs, miRNA seeds, candidate regulation)
    ("regulatory_region", "candidate_binding", "regulator"): "inferred_candidate",
    ("regulator", "rev_candidate_binding", "regulatory_region"): "inferred_candidate",
    ("regulator", "candidate_regulation", "gene"): "inferred_candidate",
    ("gene", "rev_candidate_regulation", "regulator"): "inferred_candidate",
    ("cpa_factor", "modulates_apa", "gene"): "inferred_candidate",
    ("gene", "rev_modulates_apa", "cpa_factor"): "inferred_candidate",
}

NODE_FEATURE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "population": (
        "log1p_n_cells",
        "meets_min_cells",
        "mean_shortening_index",
        "mean_magnetization_genomic",
        "mean_apa_entropy",
    ),
    "gene": (
        "log1p_mean_gene_umi",
        "n_pas_sites",
        "n_utr_isoforms",
        "shortening_index",
        "magnetization_genomic",
        "apa_entropy",
    ),
    "pas": (
        "relative_coord_norm",
        "strand_sign",
        "pas_proportion",
        "log1p_pas_umi",
        "usage_ci_width",
    ),
    "utr_isoform": (
        "log1p_sequence_length",
        "model_weight",
        "pas_proportion",
        "is_unique_model",
        "is_model_dependent",
        "is_unresolved_model",
        "is_suspicious_span",
    ),
    "regulatory_region": (
        "relative_region_pos",
        "region_length_bp",
        "accessibility",
        "survival_probability",
        "is_autr_region",
        "evidence_factor",
    ),
    "regulator": (
        "log1p_mean_umi_per_cell",
        "abundance_factor",
        "is_rbp",
        "is_mirna",
        "max_evidence_factor",
        "fractional_cis_dosage_total",
        "fractional_cis_dosage_autr",
    ),
    "cpa_factor": (
        "log1p_mean_expr",
        "pct_expressing",
    ),
}

EDGE_FEATURE_SCHEMAS: dict[str, tuple[str, ...]] = {
    "structural": (
        "structural_constant",
        "evidence_factor",
    ),
    "annotation_derived": (
        "model_weight",
        "is_ambiguous_model",
        "is_unresolved_model",
        "evidence_factor",
    ),
    "observed": (
        "observed_weight",
        "log1p_umi_or_count",
        "ci_width",
        "evidence_factor",
    ),
    "inferred_candidate": (
        "sequence_or_cis_prior",
        "accessibility",
        "evidence_factor",
        "interaction_score",
        "is_complete_prediction",
    ),
}

EVIDENCE_TIER_TO_ID: dict[str, int] = {
    "unknown": 0,
    "motif_heuristic": 1,
    "computational_prediction": 2,
    "external_database": 3,
    "experimental_other_context": 4,
    "experimental_matching_context": 5,
    "sequence_or_import": 6,
    "annotation_model": 7,
    "observed_reference": 8,
    "Sierra_UMI": 9,
    "curated_UMI": 10,
}

ID_TO_EVIDENCE_TIER: dict[int, str] = {v: k for k, v in EVIDENCE_TIER_TO_ID.items()}

EVIDENCE_TIER_FACTORS: dict[str, float] = {
    "unknown": 0.0,
    "motif_heuristic": 0.25,
    "computational_prediction": 0.35,
    "external_database": 0.55,
    "experimental_other_context": 0.75,
    "experimental_matching_context": 1.0,
    "sequence_or_import": 1.0,
    "annotation_model": 1.0,
    "observed_reference": 1.0,
    "Sierra_UMI": 1.0,
    "curated_UMI": 1.0,
}

AMBIGUITY_STATUS_TO_CODE: dict[str, int] = {
    "unique": 0,
    "ambiguous_multiple_models": 1,
    "model_dependent": 1,
    "no_compatible_3utr_model": 2,
}

CODE_TO_AMBIGUITY_STATUS: dict[int, str] = {
    0: "unique",
    1: "ambiguous_multiple_models",
    2: "no_compatible_3utr_model",
}

SPAN_STATUS_TO_CODE: dict[str, int] = {
    "within_span_limit": 0,
    "suspicious_gene_pas_span": 1,
}


@dataclass(frozen=True)
class ValidationReport:
    """Result of validating a `HeteroData` graph against `cell-regulatory-state/v0.1`."""

    valid: bool
    schema_version: str
    node_counts: dict[str, int]
    edge_counts: dict[str, int]
    edge_categories_present: list[str]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_cell_regulatory_state(data: HeteroData, *, raise_on_error: bool = True) -> ValidationReport:
    """Validate that a PyG `HeteroData` object conforms to `cell-regulatory-state/v0.1`."""
    errors: list[str] = []
    warnings: list[str] = []

    schema_version = getattr(data, "schema_version", None)
    if schema_version != SCHEMA_VERSION:
        errors.append(f"Expected schema_version={SCHEMA_VERSION!r}, got {schema_version!r}")

    if getattr(data, "is_temporal_observation", None) is None:
        errors.append("Missing graph-level `is_temporal_observation` flag (cell types are not temporal by default)")

    node_counts: dict[str, int] = {}
    for ntype in data.node_types:
        if ntype not in NODE_TYPES:
            errors.append(f"Unrecognized node type {ntype!r}")
            continue
        store = data[ntype]
        if "x" not in store:
            errors.append(f"Node type {ntype!r} missing feature tensor `x`")
            continue
        x = store["x"]
        n_nodes, n_feats = x.shape
        node_counts[ntype] = int(n_nodes)

        expected_feats = NODE_FEATURE_SCHEMAS[ntype]
        if n_feats != len(expected_feats):
            errors.append(
                f"Node type {ntype!r} has {n_feats} features, expected {len(expected_feats)} ({expected_feats})"
            )

        if torch.isnan(x).any() or torch.isinf(x).any():
            errors.append(
                f"Node type {ntype!r} `x` contains raw NaN/Inf; missing values must use neutral fill + `x_missing_mask`"
            )

        if "x_missing_mask" not in store:
            errors.append(f"Node type {ntype!r} missing `x_missing_mask`")
        else:
            mask = store["x_missing_mask"]
            if mask.shape != x.shape or mask.dtype != torch.bool:
                errors.append(f"Node type {ntype!r} `x_missing_mask` must be bool tensor of shape {tuple(x.shape)}")

        node_ids = getattr(store, "node_ids", None)
        if node_ids is None or len(node_ids) != n_nodes:
            errors.append(f"Node type {ntype!r} missing stable `node_ids` list of length {n_nodes}")
        elif len(set(node_ids)) != len(node_ids):
            errors.append(f"Node type {ntype!r} contains duplicate `node_ids`")

        if ntype == "utr_isoform" and n_nodes > 0:
            if "ambiguity_code" not in store or "is_unresolved_mask" not in store:
                errors.append("Node type 'utr_isoform' must carry explicit `ambiguity_code` and `is_unresolved_mask`")

        if ntype == "regulator" and n_nodes > 0:
            # Verify miRNA nodes have explicit missingness on abundance (10x does not measure endogenous miRNAs)
            is_mirna = x[:, 3] > 0.5
            if is_mirna.any() and "x_missing_mask" in store:
                mirna_umi_missing = store["x_missing_mask"][is_mirna, 0]
                if not bool(mirna_umi_missing.all()):
                    errors.append(
                        "Endogenous miRNA regulator nodes must mark `log1p_mean_umi_per_cell` as missing in `x_missing_mask`"
                    )

    edge_counts: dict[str, int] = {}
    categories_seen: set[str] = set()
    for etype in data.edge_types:
        src_t, rel, dst_t = etype
        key_str = f"{src_t}__{rel}__{dst_t}"
        store = data[etype]
        if etype not in CANONICAL_EDGE_TYPES:
            errors.append(f"Unrecognized edge type {etype!r}")
            continue
        expected_cat = CANONICAL_EDGE_TYPES[etype]
        actual_cat = getattr(store, "edge_category", None)
        if actual_cat != expected_cat:
            errors.append(f"Edge type {etype} has edge_category={actual_cat!r}, expected {expected_cat!r}")
        else:
            categories_seen.add(actual_cat)

        if "edge_index" not in store:
            errors.append(f"Edge type {etype} missing `edge_index`")
            continue
        edge_index = store["edge_index"]
        n_edges = int(edge_index.shape[1])
        edge_counts[key_str] = n_edges

        if n_edges > 0:
            if int(edge_index[0].min()) < 0 or int(edge_index[0].max()) >= data[src_t].x.shape[0]:
                errors.append(f"Edge type {etype} source indices out of bounds for {src_t}")
            if int(edge_index[1].min()) < 0 or int(edge_index[1].max()) >= data[dst_t].x.shape[0]:
                errors.append(f"Edge type {etype} destination indices out of bounds for {dst_t}")

        if "edge_attr" not in store or "edge_attr_missing_mask" not in store:
            errors.append(f"Edge type {etype} missing `edge_attr` or `edge_attr_missing_mask`")
        else:
            ea = store["edge_attr"]
            em = store["edge_attr_missing_mask"]
            expected_efeats = EDGE_FEATURE_SCHEMAS[expected_cat]
            if ea.shape != (n_edges, len(expected_efeats)):
                errors.append(
                    f"Edge type {etype} `edge_attr` shape {tuple(ea.shape)} != ({n_edges}, {len(expected_efeats)})"
                )
            if em.shape != ea.shape or em.dtype != torch.bool:
                errors.append(f"Edge type {etype} `edge_attr_missing_mask` must match `edge_attr` shape and be bool")
            if torch.isnan(ea).any() or torch.isinf(ea).any():
                errors.append(f"Edge type {etype} `edge_attr` contains raw NaN/Inf")

        if "evidence_tier_id" not in store:
            errors.append(f"Edge type {etype} missing `evidence_tier_id` tensor")
        elif store["evidence_tier_id"].shape != (n_edges,) or store["evidence_tier_id"].dtype != torch.long:
            errors.append(f"Edge type {etype} `evidence_tier_id` must be 1D long tensor of length {n_edges}")

        if "is_candidate_only" not in store:
            errors.append(f"Edge type {etype} missing `is_candidate_only` boolean tensor")

    report = ValidationReport(
        valid=len(errors) == 0,
        schema_version=str(schema_version),
        node_counts=node_counts,
        edge_counts=edge_counts,
        edge_categories_present=sorted(categories_seen),
        errors=errors,
        warnings=warnings,
    )
    if errors and raise_on_error:
        raise ValueError("Invalid cell-regulatory-state/v0.1 HeteroData:\n- " + "\n- ".join(errors))
    return report
