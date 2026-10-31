"""Task C tests: Tahoe-100M sparse adapter, sample-level leakage guard, cache byte cap, and 4-model benchmark."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from utrnnet.tahoe import (
    GeneGraphDeltaPredictor,
    RegulatoryPriorHeteroDeltaPredictor,
    RidgeDeltaPredictor,
    run_four_model_tahoe_benchmark,
    run_smoke_fixture,
    sparse_counts,
    sparse_resource_report,
    validate_group_partition,
    write_sparse_cache,
    zero_delta,
)


def test_tahoe_smoke_fixture_and_sparse_cache_budget(tmp_path: Path):
    fixture_path = Path(__file__).resolve().parents[2] / "examples" / "tahoe_smoke_fixture.json"
    res = run_smoke_fixture(fixture_path)
    assert res["status"] == "pass"
    assert res["missing_modalities"] == ["PAS usage", "timepoint"]
    assert res["biological_claim"] is False

    import json

    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    gene_meta = pd.DataFrame(fixture["gene_metadata"])
    mat, cell_ids, gene_ids = sparse_counts(fixture["expression_records"], gene_meta)
    rep = sparse_resource_report(mat, elapsed_seconds=0.05)
    assert rep["shape"] == [len(cell_ids), len(gene_ids)]
    assert rep["csr_payload_bytes"] > 0

    cache_path = tmp_path / "pilot_cache.npz"
    out_npz, out_json = write_sparse_cache(mat, cell_ids, gene_ids, cache_path, max_bytes=1024 * 1024)
    assert out_npz.exists() and out_json.exists()

    with pytest.raises(ValueError, match="exceeding max_bytes"):
        write_sparse_cache(mat, cell_ids, gene_ids, tmp_path / "too_small.npz", max_bytes=8)


def test_tahoe_sample_level_partition_leakage_guard():
    valid_df = pd.DataFrame(
        {
            "sample": ["smp_1588", "smp_1588", "smp_1684", "smp_1780"],
            "cell_line_id": ["CVCL_0399", "CVCL_0459", "CVCL_0399", "CVCL_0459"],
            "partition": ["train", "train", "train", "test"],
        }
    )
    validate_group_partition(valid_df, group_column="sample", partition_column="partition")

    leaked_df = pd.DataFrame(
        {
            "sample": ["smp_1588", "smp_1588", "smp_1684"],
            "cell_line_id": ["CVCL_0399", "CVCL_0459", "CVCL_0399"],
            "partition": ["train", "test", "train"],
        }
    )
    with pytest.raises(ValueError, match="partition leakage"):
        validate_group_partition(leaked_df, group_column="sample", partition_column="partition")


def test_tahoe_four_model_benchmark_identical_splits_and_uncertainty():
    rng = np.random.default_rng(42)
    n_samples = 16
    n_genes = 12
    n_regs = 5
    ctx_dim = 6

    # 8 unique sample IDs x 2 cell lines = 16 sample-line rows; hold out last 2 samples (4 rows)
    sample_ids = [f"smp_{100 + i // 2}" for i in range(n_samples)]
    conditions = [f"cond_{i % 4}" for i in range(n_samples)]
    train_mask = np.array([i < 12 for i in range(n_samples)], dtype=bool)

    basal = rng.normal(1.5, 0.4, size=(n_samples, n_genes)).astype(np.float32)
    context = rng.normal(0.0, 1.0, size=(n_samples, ctx_dim)).astype(np.float32)

    gene_ei = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=np.int64)
    reg_ei = np.array([[0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5]], dtype=np.int64)
    reg_ew = np.array([0.8, 0.6, 0.9, 0.5, 0.7, 1.0], dtype=np.float32)
    gene_priors = rng.uniform(0.1, 1.5, size=(n_genes, 4)).astype(np.float32)

    # Construct synthetic ground-truth delta with structured signal from context + regulator prior edges
    w_true = rng.normal(0.0, 0.4, size=(ctx_dim, n_genes)).astype(np.float32)
    delta_targets = (context @ w_true - 0.15 * (basal - basal.mean(axis=0))).astype(np.float32)

    bench = run_four_model_tahoe_benchmark(
        sample_ids=sample_ids,
        conditions=conditions,
        basal_log_counts=basal,
        context_features=context,
        delta_targets=delta_targets,
        gene_edge_index=gene_ei,
        reg_to_gene_edge_index=reg_ei,
        reg_to_gene_edge_weight=reg_ew,
        gene_structural_priors=gene_priors,
        n_regulators=n_regs,
        train_mask=train_mask,
        epochs=40,
        lr=5e-3,
        top_k=4,
        seed=42,
        device="cpu",
    )

    assert bench["sample_partition_isolation_verified"] is True
    assert bench["missing_modalities"] == ["PAS usage", "timepoint"]
    for m_name in ("zero_delta", "ridge", "regularized_mlp", "gene_graph_gnn", "regulatory_prior_hetero_gnn"):
        assert m_name in bench["models"]
        m_res = bench["models"][m_name]
        assert np.isfinite(m_res["mse"])
        assert len(m_res["mse_ci95"]) == 2
        assert m_res["mse_ci95"][0] <= m_res["mse_ci95"][1]
        assert 0.0 <= m_res["top_k_de_recovery"] <= 1.0
