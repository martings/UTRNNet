"""Task B tests: Strict compatibility between upstream `utrnet-graph/v0.1` and experimental `cell-regulatory-state/v0.1`."""
from __future__ import annotations

from pathlib import Path
import pytest

from examples.make_fixture import make_fixture
from scpolaseq_utrnet.config import load_config
from scpolaseq_utrnet.pipeline import build_gene_graphs

from utrnnet.contracts import (
    SCHEMA_VERSION,
    UPSTREAM_SCHEMA_VERSION,
    validate_utrnet_graphml,
    validate_utrnet_upstream_directory,
    verify_upstream_to_cell_state_compatibility,
)


def test_upstream_utrnet_graph_v0_1_and_cell_state_v0_1_compatibility_on_synthetic(tmp_path: Path):
    cfg = load_config(make_fixture(tmp_path))
    built = build_gene_graphs(cfg)
    gene_dir = built["directory"]

    up_rep = validate_utrnet_upstream_directory(gene_dir, raise_on_error=True)
    assert up_rep.valid
    assert up_rep.upstream_schema_version == UPSTREAM_SCHEMA_VERSION

    gml_files = sorted((gene_dir / "graphs").glob("*.graphml"))
    assert len(gml_files) == 2
    for gml in gml_files:
        gml_rep = validate_utrnet_graphml(gml, raise_on_error=True)
        assert gml_rep.valid

    compat = verify_upstream_to_cell_state_compatibility(gene_dir, raise_on_error=True)
    assert compat.compatible
    assert compat.upstream_schema_version == "utrnet-graph/v0.1"
    assert compat.experimental_schema_version == "cell-regulatory-state/v0.1"
    assert all(compat.entity_conservation_checks.values())


def test_upstream_validator_detects_missing_required_table(tmp_path: Path):
    cfg = load_config(make_fixture(tmp_path))
    built = build_gene_graphs(cfg)
    gene_dir = built["directory"]

    (gene_dir / "interaction_scores.csv").unlink()
    with pytest.raises(ValueError, match="interaction_scores.csv"):
        validate_utrnet_upstream_directory(gene_dir, raise_on_error=True)


def test_external_pbmc1k_provider_compatibility_if_available():
    external_sat1 = Path("/home/megas/agents/scpolaseq_utrnet/outputs/pbmc1k/ENSG00000130066")
    if not external_sat1.exists():
        pytest.skip("External scpolaseq_utrnet PBMC1k output directory not present")

    compat = verify_upstream_to_cell_state_compatibility(external_sat1, raise_on_error=True)
    assert compat.compatible
    assert compat.experimental_report.node_counts["population"] == 16
    assert compat.experimental_report.node_counts["gene"] == 1
    assert compat.experimental_report.node_counts["pas"] == 26
    assert compat.experimental_report.node_counts["utr_isoform"] == 34
