"""Stable upstream interface contract (`utrnet-graph/v0.1`) and compatibility verifier with `cell-regulatory-state/v0.1`."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import networkx as nx
import pandas as pd

from utrnnet.contracts.schema import (
    CANONICAL_EDGE_TYPES,
    EVIDENCE_TIER_TO_ID,
    SCHEMA_VERSION,
    ValidationReport,
    validate_cell_regulatory_state,
)
from utrnnet.graphs.adapter import from_graphml, from_utrnet_directory

UPSTREAM_SCHEMA_VERSION = "utrnet-graph/v0.1"

REQUIRED_UPSTREAM_TABLES: dict[str, tuple[str, ...]] = {
    "apa_profiles.csv": (
        "gene_id",
        "celltype_corrected",
        "site_id",
        "umi_sum",
        "total_gene_umi",
        "pas_proportion",
        "n_cells",
    ),
    "utr_isoforms.csv": (
        "isoform_id",
        "gene_id",
        "site_id",
        "sequence_length",
        "ambiguity_status",
        "model_weight",
    ),
    "regulatory_sites.csv": (
        "regulatory_site_id",
        "isoform_id",
        "site_id",
        "regulator",
        "regulator_class",
        "genomic_start",
        "genomic_end",
        "accessibility",
        "evidence_tier",
    ),
    "regulator_abundance.csv": (
        "population",
        "regulator",
        "mean_umi_per_cell",
    ),
    "interaction_scores.csv": (
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
    ),
}

OPTIONAL_UPSTREAM_TABLES: dict[str, tuple[str, ...]] = {
    "pas_sites.csv": ("site_id", "chr", "start", "end", "strand", "gene_id"),
    "cpa_regulatory_edges.csv": ("source", "factor", "correlation_r", "p_value", "evidence_tier"),
    "cpa_expression.csv": ("population", "factor", "mean_expr", "pct_expressing"),
}

UPSTREAM_TO_EXPERIMENTAL_NODE_MAP: dict[str, str] = {
    "population": "population",
    "gene": "gene",
    "PAS": "pas",
    "UTR_isoform": "utr_isoform",
    "regulatory_region": "regulatory_region",
    "RBP": "regulator",
    "ARE-BP": "regulator",
    "miRNA": "regulator",
    "cpa_factor": "cpa_factor",
}

UPSTREAM_TO_EXPERIMENTAL_EDGE_MAP: dict[str, tuple[str, str, str]] = {
    "gene_has_PAS": ("gene", "has_pas", "pas"),
    "PAS_defines_candidate_isoform": ("pas", "defines_isoform", "utr_isoform"),
    "contains_site": ("utr_isoform", "contains_site", "regulatory_region"),
    "uses_isoform": ("population", "uses_isoform", "utr_isoform"),
    "expresses_gene": ("population", "expresses_gene", "gene"),
    "expresses_regulator": ("population", "expresses_regulator", "regulator"),
    "candidate_binding": ("regulatory_region", "candidate_binding", "regulator"),
    "candidate_regulation": ("regulator", "candidate_regulation", "gene"),
    "modulates_apa": ("cpa_factor", "modulates_apa", "gene"),
}


@dataclass(frozen=True)
class UpstreamValidationReport:
    """Validation report for a `utrnet-graph/v0.1` gene output directory or GraphML artifact."""

    valid: bool
    upstream_schema_version: str
    source_path: str
    present_tables: list[str]
    row_counts: dict[str, int]
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SchemaCompatibilityReport:
    """Strict bidirectional compatibility check between `utrnet-graph/v0.1` and `cell-regulatory-state/v0.1`."""

    compatible: bool
    upstream_schema_version: str
    experimental_schema_version: str
    upstream_report: UpstreamValidationReport
    experimental_report: ValidationReport
    entity_conservation_checks: dict[str, bool]
    errors: list[str] = field(default_factory=list)


def validate_utrnet_upstream_directory(
    gene_dir: Path | str,
    *,
    raise_on_error: bool = True,
) -> UpstreamValidationReport:
    """Validate that a directory produced by `scpolaseq_utrnet` conforms to `utrnet-graph/v0.1`."""
    gene_dir = Path(gene_dir)
    errors: list[str] = []
    warnings: list[str] = []
    present_tables: list[str] = []
    row_counts: dict[str, int] = {}

    if not gene_dir.exists() or not gene_dir.is_dir():
        errors.append(f"Upstream directory does not exist: {gene_dir}")
        report = UpstreamValidationReport(
            valid=False,
            upstream_schema_version=UPSTREAM_SCHEMA_VERSION,
            source_path=str(gene_dir),
            present_tables=[],
            row_counts={},
            errors=errors,
            warnings=warnings,
        )
        if raise_on_error:
            raise ValueError("\n".join(errors))
        return report

    for table_name, req_cols in REQUIRED_UPSTREAM_TABLES.items():
        fpath = gene_dir / table_name
        if not fpath.exists():
            errors.append(f"Missing required upstream table {table_name!r} in {gene_dir}")
            continue
        present_tables.append(table_name)
        df_head = pd.read_csv(fpath, nrows=5)
        missing_cols = [c for c in req_cols if c not in df_head.columns]
        if missing_cols:
            errors.append(f"Table {table_name!r} missing required columns: {missing_cols}")
        # Count rows lightweightly
        with fpath.open("r", encoding="utf-8") as fh:
            n_rows = max(sum(1 for _ in fh) - 1, 0)
        row_counts[table_name] = n_rows
        if n_rows == 0 and table_name in ("apa_profiles.csv", "utr_isoforms.csv"):
            errors.append(f"Required core upstream table {table_name!r} has 0 rows")

    for opt_name, opt_cols in OPTIONAL_UPSTREAM_TABLES.items():
        fpath = gene_dir / opt_name
        if fpath.exists():
            present_tables.append(opt_name)
            df_head = pd.read_csv(fpath, nrows=5)
            missing_cols = [c for c in opt_cols if c not in df_head.columns]
            if missing_cols:
                warnings.append(f"Optional table {opt_name!r} missing columns: {missing_cols}")
            with fpath.open("r", encoding="utf-8") as fh:
                row_counts[opt_name] = max(sum(1 for _ in fh) - 1, 0)

    report = UpstreamValidationReport(
        valid=len(errors) == 0,
        upstream_schema_version=UPSTREAM_SCHEMA_VERSION,
        source_path=str(gene_dir),
        present_tables=sorted(present_tables),
        row_counts=row_counts,
        errors=errors,
        warnings=warnings,
    )
    if errors and raise_on_error:
        raise ValueError("Invalid utrnet-graph/v0.1 upstream directory:\n- " + "\n- ".join(errors))
    return report


def validate_utrnet_graphml(
    graphml_path: Path | str,
    *,
    raise_on_error: bool = True,
) -> UpstreamValidationReport:
    """Validate that a GraphML artifact from `scpolaseq_utrnet` conforms to `utrnet-graph/v0.1`."""
    graphml_path = Path(graphml_path)
    errors: list[str] = []
    warnings: list[str] = []
    if not graphml_path.exists():
        errors.append(f"GraphML file does not exist: {graphml_path}")
        report = UpstreamValidationReport(
            valid=False,
            upstream_schema_version=UPSTREAM_SCHEMA_VERSION,
            source_path=str(graphml_path),
            present_tables=[],
            row_counts={},
            errors=errors,
        )
        if raise_on_error:
            raise ValueError("\n".join(errors))
        return report

    g = nx.read_graphml(graphml_path)
    for nid, attrs in g.nodes(data=True):
        kind = str(attrs.get("kind", ""))
        nid_s = str(nid)
        is_regulator_node = nid_s.startswith("regulator:")
        if kind not in UPSTREAM_TO_EXPERIMENTAL_NODE_MAP and not is_regulator_node:
            errors.append(f"Node {nid!r} has unrecognized upstream kind={kind!r}")

    for u, v, eattrs in g.edges(data=True):
        ekind = str(eattrs.get("kind", ""))
        if ekind and ekind not in UPSTREAM_TO_EXPERIMENTAL_EDGE_MAP:
            warnings.append(f"Edge ({u}, {v}) has non-canonical kind={ekind!r}")
        ev = str(eattrs.get("evidence", "unknown"))
        if ev not in EVIDENCE_TIER_TO_ID:
            errors.append(f"Edge ({u}, {v}) has unrecognized evidence tier {ev!r}")

    report = UpstreamValidationReport(
        valid=len(errors) == 0,
        upstream_schema_version=UPSTREAM_SCHEMA_VERSION,
        source_path=str(graphml_path),
        present_tables=[graphml_path.name],
        row_counts={"nodes": g.number_of_nodes(), "edges": g.number_of_edges()},
        errors=errors,
        warnings=warnings,
    )
    if errors and raise_on_error:
        raise ValueError("Invalid utrnet-graph/v0.1 GraphML:\n- " + "\n- ".join(errors))
    return report


def verify_upstream_to_cell_state_compatibility(
    gene_dir: Path | str,
    *,
    raise_on_error: bool = True,
) -> SchemaCompatibilityReport:
    """Verify strict bidirectional schema compatibility between `utrnet-graph/v0.1` and `cell-regulatory-state/v0.1`."""
    gene_dir = Path(gene_dir)
    up_report = validate_utrnet_upstream_directory(gene_dir, raise_on_error=False)
    errors: list[str] = list(up_report.errors)

    data = from_utrnet_directory(gene_dir)
    exp_report = validate_cell_regulatory_state(data, raise_on_error=False)
    errors.extend(exp_report.errors)

    profiles = pd.read_csv(gene_dir / "apa_profiles.csv")
    isoforms = pd.read_csv(gene_dir / "utr_isoforms.csv")
    scores = pd.read_csv(
        gene_dir / "interaction_scores.csv",
        usecols=["regulatory_site_id", "regulator", "evidence_tier"],
    )

    expected_pops = int(profiles["celltype_corrected"].nunique())
    expected_pas = int(profiles["site_id"].nunique())
    active_sites = set(profiles["site_id"].unique())
    expected_iso = int(isoforms[isoforms["site_id"].isin(active_sites)]["isoform_id"].nunique())
    expected_reg_regions = int(scores["regulatory_site_id"].nunique()) if not scores.empty else 0
    expected_regulators = int(scores["regulator"].nunique()) if not scores.empty else 0

    checks = {
        "population_count_matches": exp_report.node_counts.get("population", -1) == expected_pops,
        "gene_count_is_one": exp_report.node_counts.get("gene", -1) == 1,
        "pas_count_matches": exp_report.node_counts.get("pas", -1) == expected_pas,
        "utr_isoform_count_matches": exp_report.node_counts.get("utr_isoform", -1) == expected_iso,
        "regulatory_region_count_matches": exp_report.node_counts.get("regulatory_region", -1) == expected_reg_regions,
        "regulator_count_matches": exp_report.node_counts.get("regulator", -1) == expected_regulators,
        "all_mapped_edges_canonical": all(et in CANONICAL_EDGE_TYPES for et in data.edge_types),
    }
    for cname, ok in checks.items():
        if not ok:
            errors.append(f"Entity conservation check failed: {cname}")

    graphml_dir = gene_dir / "graphs"
    if graphml_dir.exists():
        for gml in sorted(graphml_dir.glob("*.graphml"))[:2]:
            gml_up = validate_utrnet_graphml(gml, raise_on_error=False)
            errors.extend(gml_up.errors)
            gml_data = from_graphml(gml)
            gml_exp = validate_cell_regulatory_state(gml_data, raise_on_error=False)
            errors.extend(gml_exp.errors)

    compat = SchemaCompatibilityReport(
        compatible=len(errors) == 0,
        upstream_schema_version=UPSTREAM_SCHEMA_VERSION,
        experimental_schema_version=SCHEMA_VERSION,
        upstream_report=up_report,
        experimental_report=exp_report,
        entity_conservation_checks=checks,
        errors=errors,
    )
    if errors and raise_on_error:
        raise ValueError("Schema compatibility check failed:\n- " + "\n- ".join(errors))
    return compat
