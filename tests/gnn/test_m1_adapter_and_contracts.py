"""M1 unit and integration tests: Typed HeteroData adapter, serialization, and cell-regulatory-state/v0.1 contracts."""
from __future__ import annotations

from pathlib import Path
import pytest
import torch

from examples.make_fixture import make_fixture
from scpolaseq_utrnet.config import load_config
from scpolaseq_utrnet.pipeline import build_gene_graphs

import utrnnet
from utrnnet.contracts import (
    EDGE_CATEGORIES,
    EVIDENCE_TIER_TO_ID,
    SCHEMA_VERSION,
    validate_cell_regulatory_state,
)
from utrnnet.graphs import (
    from_graphml,
    from_utrnet_directory,
    load_hetero_data,
    merge_hetero_graphs,
    save_hetero_data,
)


def test_independent_package_import():
    assert utrnnet.__version__ == "0.1.0"
    assert SCHEMA_VERSION == "cell-regulatory-state/v0.1"


def test_synthetic_fixture_to_heterodata_and_roundtrip(tmp_path: Path):
    cfg = load_config(make_fixture(tmp_path))
    built = build_gene_graphs(cfg)
    gene_dir = built["directory"]

    data = from_utrnet_directory(gene_dir)
    report = validate_cell_regulatory_state(data, raise_on_error=True)
    assert report.valid
    assert report.schema_version == "cell-regulatory-state/v0.1"
    assert data.is_temporal_observation is False

    # Check separate edge categories exist and are never conflated
    assert set(report.edge_categories_present) == set(EDGE_CATEGORIES)
    assert data[("gene", "has_pas", "pas")].edge_category == "structural"
    assert data[("pas", "defines_isoform", "utr_isoform")].edge_category == "annotation_derived"
    assert data[("population", "uses_isoform", "utr_isoform")].edge_category == "observed"
    assert data[("regulatory_region", "candidate_binding", "regulator")].edge_category == "inferred_candidate"
    assert bool(data[("regulatory_region", "candidate_binding", "regulator")].is_candidate_only.all())
    assert not bool(data[("gene", "has_pas", "pas")].is_candidate_only.any())

    # Check stable identifiers
    assert data["population"].node_ids == ["population:A", "population:B"]
    assert data["gene"].node_ids == ["gene:ENSGPLUS"]
    assert data["pas"].node_ids == ["pas:plus:125", "pas:plus:140"]
    assert set(data["regulator"].node_ids) == {"regulator:ELAVL1", "regulator:hsa-miR-fixture"}

    # Check explicit missingness for endogenous miRNA abundance
    mir_idx = data["regulator"].node_id_to_index["regulator:hsa-miR-fixture"]
    rbp_idx = data["regulator"].node_id_to_index["regulator:ELAVL1"]
    assert bool(data["regulator"].x_missing_mask[mir_idx, 0]) is True
    assert bool(data["regulator"].x_missing_mask[mir_idx, 1]) is True
    assert bool(data["regulator"].x_missing_mask[rbp_idx, 0]) is False

    # Check evidence tiers survive conversion
    bind_tiers = data[("regulatory_region", "candidate_binding", "regulator")].evidence_tier_id.tolist()
    assert EVIDENCE_TIER_TO_ID["motif_heuristic"] in bind_tiers
    assert EVIDENCE_TIER_TO_ID["computational_prediction"] in bind_tiers

    # Serialization round-trip test
    out_pt = tmp_path / "serialized" / "state.pt"
    save_hetero_data(data, out_pt)
    assert out_pt.exists()
    assert out_pt.with_suffix(".pt.meta.json").exists()

    reloaded = load_hetero_data(out_pt)
    validate_cell_regulatory_state(reloaded, raise_on_error=True)

    for ntype in data.node_types:
        assert reloaded[ntype].node_ids == data[ntype].node_ids
        assert reloaded[ntype].node_id_to_index == data[ntype].node_id_to_index
        assert torch.allclose(reloaded[ntype].x, data[ntype].x)
        assert torch.equal(reloaded[ntype].x_missing_mask, data[ntype].x_missing_mask)

    for etype in data.edge_types:
        assert reloaded[etype].edge_category == data[etype].edge_category
        assert torch.equal(reloaded[etype].edge_index, data[etype].edge_index)
        assert torch.allclose(reloaded[etype].edge_attr, data[etype].edge_attr)
        assert torch.equal(reloaded[etype].edge_attr_missing_mask, data[etype].edge_attr_missing_mask)
        assert torch.equal(reloaded[etype].evidence_tier_id, data[etype].evidence_tier_id)
        assert reloaded[etype].evidence_tier_labels == data[etype].evidence_tier_labels


def test_graphml_conversion_and_isoform_ambiguity_preservation(tmp_path: Path):
    cfg = load_config(make_fixture(tmp_path))
    built = build_gene_graphs(cfg)
    graphml_files = sorted((built["directory"] / "graphs").glob("*.graphml"))
    assert len(graphml_files) == 2

    g_data = from_graphml(graphml_files[0])
    report = validate_cell_regulatory_state(g_data, raise_on_error=True)
    assert report.valid
    assert g_data["gene"].node_ids == ["gene:ENSGPLUS"]
    assert "regulator:hsa-miR-fixture" in g_data["regulator"].node_ids
    mir_idx = g_data["regulator"].node_id_to_index["regulator:hsa-miR-fixture"]
    assert bool(g_data["regulator"].x_missing_mask[mir_idx, 0]) is True

    # Merge two population graphs and verify deduplication of shared nodes
    g_data_2 = from_graphml(graphml_files[1])
    merged = merge_hetero_graphs([g_data, g_data_2])
    assert validate_cell_regulatory_state(merged, raise_on_error=True).valid
    assert len(merged["population"].node_ids) == 2
    assert len(merged["gene"].node_ids) == 1


def test_schema_validator_rejects_raw_nan_and_unmasked_mirna(tmp_path: Path):
    cfg = load_config(make_fixture(tmp_path))
    built = build_gene_graphs(cfg)
    data = from_utrnet_directory(built["directory"])

    bad_nan = data.clone()
    bad_nan["gene"].x = bad_nan["gene"].x.clone()
    bad_nan["gene"].x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="raw NaN/Inf"):
        validate_cell_regulatory_state(bad_nan, raise_on_error=True)

    bad_mirna = data.clone()
    bad_mirna["regulator"].x_missing_mask = bad_mirna["regulator"].x_missing_mask.clone()
    bad_mirna["regulator"].x_missing_mask[:, 0] = False
    with pytest.raises(ValueError, match="Endogenous miRNA"):
        validate_cell_regulatory_state(bad_mirna, raise_on_error=True)
