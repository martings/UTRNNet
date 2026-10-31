"""Independent explanation grounding evaluator and UTRNet contrast packet builder."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from utrnnet.interpretation.provider import (
    EvidenceItem,
    StructuredEvidencePacket,
    StructuredInterpretationResponse,
    _classify_direction,
)


@dataclass(frozen=True)
class GroundingEvaluationMetrics:
    """Explanation grounding metrics measured strictly independently from GNN predictive performance."""

    scenario_id: str
    provider_name: str
    model_id: str
    evidence_reference_precision: float
    evidence_reference_recall: float
    direction_accuracy: float
    epistemic_separation_accuracy: float
    uncertainty_calibration_accuracy: float
    unsupported_claim_detection_rate: float
    hallucinated_reference_count: int
    latency_seconds: float


def evaluate_interpretation_grounding(
    packet: StructuredEvidencePacket,
    response: StructuredInterpretationResponse,
) -> GroundingEvaluationMetrics:
    """Score a `StructuredInterpretationResponse` against its input `StructuredEvidencePacket`."""
    valid_items = {item.evidence_id: item for item in packet.evidence_items if not item.is_missing}
    missing_items = {item.evidence_id: item for item in packet.evidence_items if item.is_missing}
    all_known_ids = set(valid_items.keys()) | set(missing_items.keys())

    cited_refs: list[str] = []
    for c in response.claims:
        cited_refs.extend(c.evidence_refs)

    if cited_refs:
        valid_hits = sum(1 for r in cited_refs if r in valid_items)
        ref_precision = float(valid_hits / len(cited_refs))
        hallucinated_refs = sum(1 for r in cited_refs if r not in all_known_ids)
    else:
        ref_precision = 1.0 if not valid_items else 0.0
        hallucinated_refs = 0

    unique_valid_cited = {r for r in cited_refs if r in valid_items}
    ref_recall = float(len(unique_valid_cited) / len(valid_items)) if valid_items else 1.0

    dir_correct = 0
    epistemic_correct = 0
    uncertainty_correct = 0
    evaluated_claims = 0

    for c in response.claims:
        matched_id = next((r for r in c.evidence_refs if r in valid_items), None)
        if matched_id is None:
            continue
        item = valid_items[matched_id]
        evaluated_claims += 1
        expected_dir = _classify_direction(item.delta)
        if c.change_direction == expected_dir:
            dir_correct += 1
        if c.epistemic_status == item.epistemic_category:
            epistemic_correct += 1
        # Calibration check: candidate_prior, predicted, or non-unique ambiguity must not claim 'low' uncertainty
        requires_elevated_uncertainty = (
            item.is_candidate_only
            or item.epistemic_category in ("predicted", "candidate_prior")
            or item.ambiguity_status != "unique"
        )
        if requires_elevated_uncertainty:
            if c.uncertainty_level in ("moderate", "high"):
                uncertainty_correct += 1
        else:
            if c.uncertainty_level in ("low", "moderate"):
                uncertainty_correct += 1

    dir_acc = float(dir_correct / evaluated_claims) if evaluated_claims > 0 else 0.0
    epi_acc = float(epistemic_correct / evaluated_claims) if evaluated_claims > 0 else 0.0
    unc_acc = float(uncertainty_correct / evaluated_claims) if evaluated_claims > 0 else 0.0

    expected_unsupported_total = len(missing_items) + len(packet.candidate_trap_prompts)
    if expected_unsupported_total == 0:
        unsupported_rate = 1.0
    else:
        # Check how many missing items were NOT falsely included as supported claims, plus rejected count
        falsely_supported_missing = sum(
            1 for c in response.claims if any(r in missing_items for r in c.evidence_refs) and c.is_supported
        )
        detected = min(len(response.rejected_unsupported_claims), expected_unsupported_total)
        unsupported_rate = max(
            0.0,
            float((detected - falsely_supported_missing) / expected_unsupported_total),
        )

    return GroundingEvaluationMetrics(
        scenario_id=packet.scenario_id,
        provider_name=response.provider_name,
        model_id=response.model_id,
        evidence_reference_precision=ref_precision,
        evidence_reference_recall=ref_recall,
        direction_accuracy=dir_acc,
        epistemic_separation_accuracy=epi_acc,
        uncertainty_calibration_accuracy=unc_acc,
        unsupported_claim_detection_rate=unsupported_rate,
        hallucinated_reference_count=hallucinated_refs,
        latency_seconds=response.raw_latency_seconds,
    )


def build_evidence_packet_from_utrnet_contrast(
    gene_dir: Path | str,
    pop_a: str,
    pop_b: str,
    *,
    gene_symbol: str | None = None,
    predicted_gene_delta: float | None = None,
) -> StructuredEvidencePacket:
    """Construct a `StructuredEvidencePacket` from an observed UTRNet two-population contrast."""
    gene_dir = Path(gene_dir)
    profiles = pd.read_csv(gene_dir / "apa_profiles.csv")
    scores = pd.read_csv(
        gene_dir / "interaction_scores.csv",
        usecols=[
            "population",
            "regulatory_site_id",
            "regulator",
            "regulator_class",
            "evidence_tier",
            "interaction_score",
            "score_status",
        ],
    )
    gene_id = str(profiles["gene_id"].iloc[0]).split(".")[0]
    sym = gene_symbol or gene_dir.name

    sub_a = profiles[profiles["celltype_corrected"] == pop_a]
    sub_b = profiles[profiles["celltype_corrected"] == pop_b]
    if sub_a.empty or sub_b.empty:
        raise ValueError(f"Populations {pop_a!r} and/or {pop_b!r} not found in {gene_dir}")

    if "si" in sub_a.columns and "si" in sub_b.columns and sub_a["si"].notna().any():
        si_a = float(sub_a["si"].iloc[0])
        si_b = float(sub_b["si"].iloc[0])
        si_metric = "shortening_index"
    else:
        si_a = float(sub_a["pas_proportion"].iloc[0])
        si_b = float(sub_b["pas_proportion"].iloc[0])
        si_metric = "proximal_pas_proportion"

    if (
        "magnetization_genomic" in sub_a.columns
        and "magnetization_genomic" in sub_b.columns
        and sub_a["magnetization_genomic"].notna().any()
    ):
        mag_a = float(sub_a["magnetization_genomic"].iloc[0])
        mag_b = float(sub_b["magnetization_genomic"].iloc[0])
        mag_metric = "magnetization_genomic"
    else:
        mag_a = float(sub_a["total_gene_umi"].iloc[0])
        mag_b = float(sub_b["total_gene_umi"].iloc[0])
        mag_metric = "total_gene_umi"

    items: list[EvidenceItem] = [
        EvidenceItem(
            evidence_id=f"obs:{gene_id}:{si_metric}:{pop_a}_vs_{pop_b}",
            entity_type="gene_apa_state",
            source_id=f"population:{pop_a}",
            target_id=f"gene:{gene_id}",
            epistemic_category="observed",
            evidence_tier="Sierra_UMI",
            metric_name=si_metric,
            value_before=round(si_a, 4),
            value_after=round(si_b, 4),
            delta=round(si_b - si_a, 4),
            is_missing=False,
            is_candidate_only=False,
            ambiguity_status="unique",
        ),
        EvidenceItem(
            evidence_id=f"obs:{gene_id}:{mag_metric}:{pop_a}_vs_{pop_b}",
            entity_type="gene_apa_state",
            source_id=f"population:{pop_a}",
            target_id=f"gene:{gene_id}",
            epistemic_category="observed",
            evidence_tier="Sierra_UMI",
            metric_name=mag_metric,
            value_before=round(mag_a, 4),
            value_after=round(mag_b, 4),
            delta=round(mag_b - mag_a, 4),
            is_missing=False,
            is_candidate_only=False,
            ambiguity_status="unique",
        ),
    ]

    if predicted_gene_delta is not None:
        items.append(
            EvidenceItem(
                evidence_id=f"pred:{gene_id}:gnn_expression_delta:{pop_a}_vs_{pop_b}",
                entity_type="gnn_state_prediction",
                source_id=f"population:{pop_b}",
                target_id=f"gene:{gene_id}",
                epistemic_category="predicted",
                evidence_tier="computational_prediction",
                metric_name="predicted_log1p_gene_umi_delta",
                value_before=None,
                value_after=None,
                delta=round(float(predicted_gene_delta), 4),
                is_missing=False,
                is_candidate_only=True,
                ambiguity_status="unique",
            )
        )

    # Top RBP candidate regulator delta between pop_a and pop_b
    rbp_scores = scores[scores["regulator_class"] == "RBP"]
    if not rbp_scores.empty:
        agg = (
            rbp_scores[rbp_scores["population"].isin([pop_a, pop_b])]
            .groupby(["regulator", "population", "evidence_tier"], as_index=False)["interaction_score"]
            .sum()
        )
        piv = agg.pivot(index=["regulator", "evidence_tier"], columns="population", values="interaction_score").dropna()
        if not piv.empty and pop_a in piv.columns and pop_b in piv.columns:
            piv["delta"] = piv[pop_b] - piv[pop_a]
            top_row = piv.iloc[piv["delta"].abs().argmax()]
            top_reg, top_tier = top_row.name
            items.append(
                EvidenceItem(
                    evidence_id=f"prior:{gene_id}:rbp_candidate:{top_reg}",
                    entity_type="candidate_regulation",
                    source_id=f"regulator:{top_reg}",
                    target_id=f"gene:{gene_id}",
                    epistemic_category="candidate_prior",
                    evidence_tier=str(top_tier),
                    metric_name="candidate_interaction_score",
                    value_before=round(float(top_row[pop_a]), 4),
                    value_after=round(float(top_row[pop_b]), 4),
                    delta=round(float(top_row["delta"]), 4),
                    is_missing=False,
                    is_candidate_only=True,
                    ambiguity_status="model_dependent",
                )
            )

    # Unmeasured endogenous miRNA entry (explicit missingness trap)
    mir_scores = scores[scores["regulator_class"].str.lower().str.contains("mirna", na=False)]
    if not mir_scores.empty:
        mir_name = str(mir_scores["regulator"].iloc[0])
        items.append(
            EvidenceItem(
                evidence_id=f"missing:{gene_id}:mirna_abundance:{mir_name}",
                entity_type="regulator_abundance",
                source_id=f"regulator:{mir_name}",
                target_id=f"gene:{gene_id}",
                epistemic_category="candidate_prior",
                evidence_tier="computational_prediction",
                metric_name="endogenous_mirna_umi_abundance",
                value_before=None,
                value_after=None,
                delta=None,
                is_missing=True,
                is_candidate_only=True,
                ambiguity_status="model_dependent",
            )
        )

    traps = [
        f"Direct CLIP-seq physical occupancy proves {sym} 3'UTR shortening is causally driven by endogenous miRNA UMI knockdown between {pop_a} and {pop_b}.",
    ]
    return StructuredEvidencePacket(
        scenario_id=f"utrnet_contrast:{sym}:{pop_a}_vs_{pop_b}",
        gene_id=gene_id,
        gene_symbol=sym,
        contrast_description=f"Observed UTRNet state contrast for {sym} ({gene_id}) between {pop_a} and {pop_b}",
        evidence_items=items,
        candidate_trap_prompts=traps,
    )
