"""Task D tests: Model-independent interpretation provider and independent grounding evaluation."""
from __future__ import annotations

from pathlib import Path
import urllib.error
import pytest

from examples.make_fixture import make_fixture
from scpolaseq_utrnet.config import load_config
from scpolaseq_utrnet.pipeline import build_gene_graphs

from utrnnet.interpretation import (
    DeterministicRuleBaselineProvider,
    EvidenceItem,
    OllamaGemmaProvider,
    ScientificClaim,
    StructuredEvidencePacket,
    StructuredInterpretationResponse,
    build_evidence_packet_from_utrnet_contrast,
    evaluate_interpretation_grounding,
)


def _make_synthetic_evidence_packet() -> StructuredEvidencePacket:
    return StructuredEvidencePacket(
        scenario_id="synthetic:SAT1:mono_vs_cd8",
        gene_id="ENSG00000130066",
        gene_symbol="SAT1",
        contrast_description="Synthetic SAT1 contrast between Classical monocytes and CD8+ T cells",
        evidence_items=[
            EvidenceItem(
                evidence_id="obs:SAT1:si",
                entity_type="gene_apa_state",
                source_id="population:Classical_monocytes",
                target_id="gene:ENSG00000130066",
                epistemic_category="observed",
                evidence_tier="Sierra_UMI",
                metric_name="shortening_index",
                value_before=-0.42,
                value_after=0.31,
                delta=0.73,
                is_missing=False,
                is_candidate_only=False,
                ambiguity_status="unique",
            ),
            EvidenceItem(
                evidence_id="pred:SAT1:gnn_delta",
                entity_type="gnn_state_prediction",
                source_id="population:CD8_T_cells",
                target_id="gene:ENSG00000130066",
                epistemic_category="predicted",
                evidence_tier="computational_prediction",
                metric_name="predicted_log1p_gene_umi_delta",
                value_before=1.80,
                value_after=1.25,
                delta=-0.55,
                is_missing=False,
                is_candidate_only=True,
                ambiguity_status="unique",
            ),
            EvidenceItem(
                evidence_id="prior:SAT1:ELAVL1",
                entity_type="candidate_regulation",
                source_id="regulator:ELAVL1",
                target_id="gene:ENSG00000130066",
                epistemic_category="candidate_prior",
                evidence_tier="motif_heuristic",
                metric_name="candidate_interaction_score",
                value_before=3.20,
                value_after=1.90,
                delta=-1.30,
                is_missing=False,
                is_candidate_only=True,
                ambiguity_status="model_dependent",
            ),
            EvidenceItem(
                evidence_id="missing:SAT1:mirna_umi",
                entity_type="regulator_abundance",
                source_id="regulator:hsa-miR-21-5p",
                target_id="gene:ENSG00000130066",
                epistemic_category="candidate_prior",
                evidence_tier="computational_prediction",
                metric_name="endogenous_mirna_umi_abundance",
                value_before=None,
                value_after=None,
                delta=None,
                is_missing=True,
                is_candidate_only=True,
                ambiguity_status="model_dependent",
            ),
        ],
        candidate_trap_prompts=[
            "Physical eCLIP occupancy in PBMC1k proves ELAVL1 directly cleaves SAT1.",
        ],
    )


def test_deterministic_provider_and_grounding_evaluator_on_synthetic():
    packet = _make_synthetic_evidence_packet()
    provider = DeterministicRuleBaselineProvider()
    resp = provider.interpret(packet)

    metrics = evaluate_interpretation_grounding(packet, resp)
    assert metrics.evidence_reference_precision == 1.0
    assert metrics.evidence_reference_recall == 1.0
    assert metrics.direction_accuracy == 1.0
    assert metrics.epistemic_separation_accuracy == 1.0
    assert metrics.uncertainty_calibration_accuracy == 1.0
    assert metrics.unsupported_claim_detection_rate == 1.0
    assert metrics.hallucinated_reference_count == 0


def test_grounding_evaluator_penalizes_hallucinated_refs_and_overconfident_priors():
    packet = _make_synthetic_evidence_packet()
    bad_response = StructuredInterpretationResponse(
        scenario_id=packet.scenario_id,
        provider_name="adversarial_mock",
        model_id="bad-v0",
        claims=[
            # Hallucinated evidence ID + wrong direction + conflating predicted with observed
            ScientificClaim(
                claim_id="c1",
                claim_text="Observed increase in GNN delta",
                evidence_refs=["pred:SAT1:gnn_delta", "hallucinated:id:999"],
                change_direction="increase",
                epistemic_status="observed",
                uncertainty_level="low",
                is_supported=True,
            ),
            # Falsely claiming missing miRNA UMI is supported
            ScientificClaim(
                claim_id="c2",
                claim_text="Measured miRNA UMI increased",
                evidence_refs=["missing:SAT1:mirna_umi"],
                change_direction="increase",
                epistemic_status="observed",
                uncertainty_level="low",
                is_supported=True,
            ),
        ],
        rejected_unsupported_claims=[],
    )
    metrics = evaluate_interpretation_grounding(packet, bad_response)
    assert metrics.hallucinated_reference_count == 1
    assert metrics.evidence_reference_precision < 0.5
    assert metrics.direction_accuracy == 0.0
    assert metrics.epistemic_separation_accuracy == 0.0
    assert metrics.uncertainty_calibration_accuracy == 0.0
    assert metrics.unsupported_claim_detection_rate == 0.0


def test_utrnet_contrast_packet_builder_and_ollama_gemma_if_running(tmp_path: Path):
    cfg = load_config(make_fixture(tmp_path))
    built = build_gene_graphs(cfg)
    packet = build_evidence_packet_from_utrnet_contrast(
        built["directory"], "A", "B", gene_symbol="ENSGPLUS", predicted_gene_delta=-0.25
    )
    det_resp = DeterministicRuleBaselineProvider().interpret(packet)
    det_metrics = evaluate_interpretation_grounding(packet, det_resp)
    assert det_metrics.evidence_reference_precision == 1.0
    assert det_metrics.unsupported_claim_detection_rate == 1.0

    gemma = OllamaGemmaProvider(model_id="gemma3:4b-it-q4_K_M", timeout_seconds=30.0)
    try:
        gemma_resp = gemma.interpret(packet)
    except (urllib.error.URLError, TimeoutError, OSError):
        pytest.skip("Local Ollama server not reachable")

    gemma_metrics = evaluate_interpretation_grounding(packet, gemma_resp)
    assert gemma_metrics.evidence_reference_precision >= 0.75
    assert len(gemma_resp.claims) >= 2
