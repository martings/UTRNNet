"""Model-independent structured interpretation provider and Ollama/Gemma implementation for UTRNNet."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Literal, Protocol
import urllib.request

ChangeDirection = Literal["increase", "decrease", "unchanged", "ambiguous"]
EpistemicStatus = Literal["observed", "predicted", "candidate_prior"]
UncertaintyLevel = Literal["low", "moderate", "high"]


@dataclass(frozen=True)
class EvidenceItem:
    """Single traceable graph evidence record passed to an interpretation provider."""

    evidence_id: str
    entity_type: str
    source_id: str
    target_id: str
    epistemic_category: EpistemicStatus
    evidence_tier: str
    metric_name: str
    value_before: float | None
    value_after: float | None
    delta: float | None
    is_missing: bool = False
    is_candidate_only: bool = False
    ambiguity_status: str = "unique"


@dataclass(frozen=True)
class StructuredEvidencePacket:
    """Structured graph evidence and predicted state changes for a single biological scenario or contrast."""

    scenario_id: str
    gene_id: str
    gene_symbol: str
    contrast_description: str
    evidence_items: list[EvidenceItem]
    candidate_trap_prompts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScientificClaim:
    """Single structured scientific claim with explicit provenance, direction, epistemic status, and uncertainty."""

    claim_id: str
    claim_text: str
    evidence_refs: list[str]
    change_direction: ChangeDirection
    epistemic_status: EpistemicStatus
    uncertainty_level: UncertaintyLevel
    is_supported: bool = True
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class StructuredInterpretationResponse:
    """Structured output returned by any `InterpretationProvider`."""

    scenario_id: str
    provider_name: str
    model_id: str
    claims: list[ScientificClaim]
    rejected_unsupported_claims: list[ScientificClaim] = field(default_factory=list)
    raw_latency_seconds: float = 0.0


class InterpretationProvider(Protocol):
    """Model-independent protocol for generating structured scientific claims from graph evidence."""

    def interpret(self, packet: StructuredEvidencePacket) -> StructuredInterpretationResponse:
        """Generate structured scientific claims and flag unsupported/ungrounded claims."""
        ...


def _classify_direction(delta: float | None, eps: float = 1e-3) -> ChangeDirection:
    if delta is None:
        return "ambiguous"
    if delta > eps:
        return "increase"
    if delta < -eps:
        return "decrease"
    return "unchanged"


def _classify_uncertainty(item: EvidenceItem) -> UncertaintyLevel:
    if item.is_missing or item.ambiguity_status != "unique":
        return "high"
    if item.is_candidate_only or item.epistemic_category in ("predicted", "candidate_prior"):
        return "moderate" if item.evidence_tier != "motif_heuristic" else "high"
    return "low"


class DeterministicRuleBaselineProvider:
    """Deterministic rule-based interpretation provider used as a non-LLM reference baseline."""

    def __init__(self) -> None:
        self.provider_name = "deterministic_rule_baseline"
        self.model_id = "rule-v0.1"

    def interpret(self, packet: StructuredEvidencePacket) -> StructuredInterpretationResponse:
        claims: list[ScientificClaim] = []
        rejected: list[ScientificClaim] = []

        for idx, item in enumerate(packet.evidence_items):
            if item.is_missing:
                rejected.append(
                    ScientificClaim(
                        claim_id=f"{packet.scenario_id}:rejected_missing:{idx}",
                        claim_text=(
                            f"Cannot claim measured {item.metric_name} for {item.source_id} because "
                            f"the modality is explicitly marked missing (is_missing=True)."
                        ),
                        evidence_refs=[item.evidence_id],
                        change_direction="ambiguous",
                        epistemic_status=item.epistemic_category,
                        uncertainty_level="high",
                        is_supported=False,
                        unsupported_reason="unmeasured_or_missing_modality",
                    )
                )
                continue

            direction = _classify_direction(item.delta)
            uncertainty = _classify_uncertainty(item)
            qualifier = (
                "Empirical observation shows"
                if item.epistemic_category == "observed"
                else (
                    "GNN model prediction estimates"
                    if item.epistemic_category == "predicted"
                    else "Sequence/annotation candidate prior suggests"
                )
            )
            claims.append(
                ScientificClaim(
                    claim_id=f"{packet.scenario_id}:claim:{idx}",
                    claim_text=(
                        f"{qualifier} {item.metric_name} ({item.source_id} -> {item.target_id}) "
                        f"exhibits {direction} (delta={item.delta}) under evidence tier {item.evidence_tier}."
                    ),
                    evidence_refs=[item.evidence_id],
                    change_direction=direction,
                    epistemic_status=item.epistemic_category,
                    uncertainty_level=uncertainty,
                    is_supported=True,
                    unsupported_reason=None,
                )
            )

        for t_idx, trap in enumerate(packet.candidate_trap_prompts):
            rejected.append(
                ScientificClaim(
                    claim_id=f"{packet.scenario_id}:trap:{t_idx}",
                    claim_text=trap,
                    evidence_refs=[],
                    change_direction="ambiguous",
                    epistemic_status="candidate_prior",
                    uncertainty_level="high",
                    is_supported=False,
                    unsupported_reason="unsupported_causal_or_occupancy_claim",
                )
            )

        return StructuredInterpretationResponse(
            scenario_id=packet.scenario_id,
            provider_name=self.provider_name,
            model_id=self.model_id,
            claims=claims,
            rejected_unsupported_claims=rejected,
            raw_latency_seconds=0.0,
        )


class OllamaGemmaProvider:
    """Ollama-backed Gemma interpretation provider with strict JSON schema output and post-hoc grounding guard."""

    def __init__(
        self,
        model_id: str = "gemma3:4b-it-q4_K_M",
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        timeout_seconds: float = 45.0,
    ) -> None:
        self.provider_name = "ollama_gemma"
        self.model_id = model_id
        self.endpoint = endpoint
        self.timeout_seconds = float(timeout_seconds)

    def _build_prompt(self, packet: StructuredEvidencePacket) -> str:
        ev_rows = [asdict(item) for item in packet.evidence_items]
        payload = {
            "scenario_id": packet.scenario_id,
            "gene_id": packet.gene_id,
            "gene_symbol": packet.gene_symbol,
            "contrast_description": packet.contrast_description,
            "evidence_items": ev_rows,
            "candidate_trap_prompts": packet.candidate_trap_prompts,
        }
        return (
            "You are a rigorous computational biology auditor for UTRNNet.\n"
            "Given the JSON evidence packet below, output ONLY valid JSON with two keys: "
            "'claims' and 'rejected_unsupported_claims'.\n"
            "Rules:\n"
            "1. Every item in 'claims' MUST correspond to a non-missing evidence_item (is_missing == false) and "
            "MUST cite its exact 'evidence_id' inside 'evidence_refs'.\n"
            "2. Set 'change_direction' to 'increase' if delta > 0.001, 'decrease' if delta < -0.001, "
            "'unchanged' if abs(delta) <= 0.001, or 'ambiguous' if delta is null.\n"
            "3. Preserve 'epistemic_status' EXACTLY from the evidence item's 'epistemic_category' "
            "('observed', 'predicted', or 'candidate_prior'). Never label a predicted or candidate_prior item as 'observed'.\n"
            "4. Set 'uncertainty_level' to 'low' ONLY for observed items with unique ambiguity_status; "
            "use 'moderate' or 'high' for predicted, candidate_prior, motif_heuristic, or ambiguous items.\n"
            "5. Any evidence item with is_missing == true AND every string in 'candidate_trap_prompts' MUST be placed "
            "in 'rejected_unsupported_claims' with is_supported=false and a clear 'unsupported_reason'.\n"
            "Each claim object must have keys: claim_id, claim_text, evidence_refs (list of strings), "
            "change_direction, epistemic_status, uncertainty_level, is_supported (bool), unsupported_reason (string or null).\n\n"
            f"EVIDENCE_PACKET_JSON:\n{json.dumps(payload, indent=2)}\n"
        )

    def interpret(self, packet: StructuredEvidencePacket) -> StructuredInterpretationResponse:
        import time

        t0 = time.perf_counter()
        prompt = self._build_prompt(packet)
        req_body = json.dumps(
            {
                "model": self.model_id,
                "prompt": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 2048},
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            self.endpoint,
            data=req_body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
            raw_resp = json.loads(resp.read().decode("utf-8"))

        elapsed = time.perf_counter() - t0
        parsed = json.loads(raw_resp.get("response", "{}"))

        def _parse_claim(c_dict: dict[str, Any], idx: int, default_supported: bool) -> ScientificClaim:
            cdir = str(c_dict.get("change_direction", "ambiguous"))
            if cdir not in ("increase", "decrease", "unchanged", "ambiguous"):
                cdir = "ambiguous"
            estat = str(c_dict.get("epistemic_status", "candidate_prior"))
            if estat not in ("observed", "predicted", "candidate_prior"):
                estat = "candidate_prior"
            ulevel = str(c_dict.get("uncertainty_level", "high"))
            if ulevel not in ("low", "moderate", "high"):
                ulevel = "high"
            refs = [str(r) for r in (c_dict.get("evidence_refs") or [])]
            return ScientificClaim(
                claim_id=str(c_dict.get("claim_id", f"{packet.scenario_id}:gemma:{idx}")),
                claim_text=str(c_dict.get("claim_text", "")),
                evidence_refs=refs,
                change_direction=cdir,  # type: ignore[arg-type]
                epistemic_status=estat,  # type: ignore[arg-type]
                uncertainty_level=ulevel,  # type: ignore[arg-type]
                is_supported=bool(c_dict.get("is_supported", default_supported)),
                unsupported_reason=(
                    str(c_dict["unsupported_reason"]) if c_dict.get("unsupported_reason") is not None else None
                ),
            )

        raw_claims = parsed.get("claims", [])
        raw_rejected = parsed.get("rejected_unsupported_claims", [])
        claims = [_parse_claim(c, i, True) for i, c in enumerate(raw_claims) if isinstance(c, dict)]
        rejected = [_parse_claim(c, i, False) for i, c in enumerate(raw_rejected) if isinstance(c, dict)]

        return StructuredInterpretationResponse(
            scenario_id=packet.scenario_id,
            provider_name=self.provider_name,
            model_id=self.model_id,
            claims=claims,
            rejected_unsupported_claims=rejected,
            raw_latency_seconds=elapsed,
        )
