"""M2 unit and integration tests: HeteroGNN forward/backward, baseline, negative controls, and leakage guards."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import torch

from examples.make_fixture import make_fixture
from scpolaseq_utrnet.config import load_config
from scpolaseq_utrnet.pipeline import build_gene_graphs

from utrnnet.evaluation import (
    evaluate_masked_reconstruction,
    run_three_way_benchmark,
    verify_zero_target_leakage,
)
from utrnnet.graphs import (
    apply_feature_mask_with_leakage_guard,
    from_utrnet_directory,
     merge_hetero_graphs,
    randomize_hetero_graph_edges,
)
from utrnnet.models import TrivialNonGraphBaseline, UTRNNetHeteroGNN
from utrnnet.training import (
    TrainingConfig,
    compute_masked_reconstruction_loss,
    set_reproducible_seed,
    train_masked_reconstruction,
)


def _build_synthetic_two_gene_graph(tmp_path: Path):
    cfg_plus = load_config(make_fixture(tmp_path))
    res_plus = build_gene_graphs(cfg_plus)
    g_plus = from_utrnet_directory(res_plus["directory"])

    cfg_minus = replace(cfg_plus, gene="MINUS")
    res_minus = build_gene_graphs(cfg_minus)
    g_minus = from_utrnet_directory(res_minus["directory"])

    return merge_hetero_graphs([g_plus, g_minus])


def test_cpu_forward_backward_and_latent_dim_64(tmp_path: Path):
    data = _build_synthetic_two_gene_graph(tmp_path)
    set_reproducible_seed(123)

    model = UTRNNetHeteroGNN(hidden_channels=64, num_layers=2, heads=4, dropout=0.0).to("cpu")
    batch = apply_feature_mask_with_leakage_guard(data, mask_ratio=0.4, seed=123)

    pred_dict, z_dict = model(batch.masked_data.to("cpu"))
    for ntype in data.node_types:
        assert z_dict[ntype].shape == (data[ntype].x.shape[0], 64)
        assert pred_dict[ntype].shape == data[ntype].x.shape

    loss, mae, per_type = compute_masked_reconstruction_loss(pred_dict, batch, torch.device("cpu"))
    assert loss.requires_grad
    assert float(loss.item()) > 0.0
    assert mae > 0.0
    assert len(per_type) > 0

    loss.backward()
    grad_norms = [
        float(p.grad.norm().item())
        for p in model.parameters()
        if p.grad is not None
    ]
    assert len(grad_norms) > 0
    assert any(g > 0.0 for g in grad_norms)

    # Test perturbation-conditioned scaffold head
    n_reg = data["regulator"].x.shape[0]
    reg_pert = torch.full((n_reg, 1), -1.0, dtype=torch.float32)
    deltas = model.predict_perturbation_delta(data.to("cpu"), reg_pert)
    assert deltas["gene"].shape == data["gene"].x.shape
    assert deltas["utr_isoform"].shape == data["utr_isoform"].x.shape


def test_target_masking_and_strict_leakage_protection(tmp_path: Path):
    data = _build_synthetic_two_gene_graph(tmp_path)
    set_reproducible_seed(7)
    model = UTRNNetHeteroGNN(hidden_channels=64, num_layers=2, heads=4, dropout=0.0).to("cpu")

    # 1. Verify zero target leakage on gene expression (col 0)
    assert verify_zero_target_leakage(
        model, data, target_node_type="gene", target_feature_idx=0, perturbation_magnitude=500.0
    )

    # 2. Verify zero target leakage on PAS usage (col 2) across coupled utr_isoform and uses_isoform edges
    assert verify_zero_target_leakage(
        model, data, target_node_type="pas", target_feature_idx=2, perturbation_magnitude=500.0
    )

    # 3. Verify zero target leakage on regulator abundance (col 0) across coupled abundance_factor and edges
    assert verify_zero_target_leakage(
        model, data, target_node_type="regulator", target_feature_idx=0, perturbation_magnitude=500.0
    )

    # 4. Verify that candidate interaction_score (col 3) is always masked during reconstruction
    batch = apply_feature_mask_with_leakage_guard(data, mask_ratio=0.3, seed=7)
    cand_bind = batch.masked_data[("regulatory_region", "candidate_binding", "regulator")]
    assert torch.all(cand_bind.edge_attr[:, 3] == 0.0)
    assert torch.all(cand_bind.edge_attr_missing_mask[:, 3])

    # 5. Verify indirect PAS-usage leakage path is guarded (survival_probability, cis-dosage, gene/pop APA summaries)
    explicit_mask = {nt: torch.zeros_like(data[nt].x_missing_mask) for nt in data.node_types}
    explicit_mask["pas"][0, 2] = True
    pas_batch = apply_feature_mask_with_leakage_guard(data, mask_ratio=0.0, explicit_node_mask=explicit_mask)
    assert torch.all(pas_batch.masked_data["regulatory_region"].x[:, 3] == 0.0)
    assert torch.all(pas_batch.masked_data["regulatory_region"].x_missing_mask[:, 3])
    assert torch.all(pas_batch.masked_data["regulator"].x[:, 5:7] == 0.0)
    assert torch.all(pas_batch.masked_data["regulator"].x_missing_mask[:, 5:7])
    assert torch.all(pas_batch.masked_data["gene"].x[:, 3:6] == 0.0)
    assert torch.all(pas_batch.masked_data["gene"].x_missing_mask[:, 3:6])
    assert torch.all(pas_batch.masked_data[("regulator", "candidate_regulation", "gene")].edge_attr[:, 0] == 0.0)
    assert torch.all(pas_batch.masked_data[("regulator", "candidate_regulation", "gene")].edge_attr_missing_mask[:, 0])


def test_negative_control_edge_randomization(tmp_path: Path):
    data = _build_synthetic_two_gene_graph(tmp_path)
    ctrl = randomize_hetero_graph_edges(data, seed=99)

    assert getattr(ctrl, "is_negative_control", False) is True
    for ntype in data.node_types:
        assert torch.allclose(ctrl[ntype].x, data[ntype].x)
        assert torch.equal(ctrl[ntype].x_missing_mask, data[ntype].x_missing_mask)

    for etype in data.edge_types:
        assert ctrl[etype].edge_index.shape == data[etype].edge_index.shape
        assert ctrl[etype].edge_category == data[etype].edge_category


def test_minimal_training_checkpoint_and_three_way_benchmark(tmp_path: Path):
    data = _build_synthetic_two_gene_graph(tmp_path)
    cfg = TrainingConfig(
        seed=42,
        hidden_channels=64,
        num_layers=2,
        heads=4,
        dropout=0.0,
        lr=5e-3,
        epochs=12,
        mask_ratio=0.3,
        device="cpu",
    )

    ckpt_dir = tmp_path / "ckpt_test"
    model = UTRNNetHeteroGNN(hidden_channels=64, num_layers=2, heads=4, dropout=0.0)
    res = train_masked_reconstruction(model, data, cfg, checkpoint_dir=ckpt_dir)

    assert res.final_loss < res.initial_loss
    assert res.checkpoint_path is not None and Path(res.checkpoint_path).exists()
    assert res.metrics_path is not None and Path(res.metrics_path).exists()

    eval_metrics = evaluate_masked_reconstruction(model, data, mask_ratio=0.3, seed=42, device="cpu")
    assert eval_metrics["mse"] <= res.final_loss
    assert eval_metrics["mse"] < res.initial_loss

    # Run three-way benchmark (baseline vs HeteroGNN vs negative-control random graph)
    bench_dir = tmp_path / "bench"
    comp = run_three_way_benchmark(data, cfg, output_dir=bench_dir)
    assert comp.leakage_audit_passed is True
    assert (bench_dir / "benchmark_summary.json").exists()
    assert comp.gnn_true_graph_final_mse > 0.0
    assert comp.baseline_final_mse > 0.0
    assert comp.gnn_random_control_final_mse > 0.0
