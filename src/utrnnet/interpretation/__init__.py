"""Structured interpretation providers and independent grounding benchmark for UTRNNet (`Task D`)."""
from .evaluator import (
    GroundingEvaluationMetrics,
    build_evidence_packet_from_utrnet_contrast,
    evaluate_interpretation_grounding,
)
from .provider import (
    DeterministicRuleBaselineProvider,
    EvidenceItem,
    InterpretationProvider,
    OllamaGemmaProvider,
    ScientificClaim,
    StructuredEvidencePacket,
    StructuredInterpretationResponse,
)

__all__ = [
    "DeterministicRuleBaselineProvider",
    "EvidenceItem",
    "GroundingEvaluationMetrics",
    "InterpretationProvider",
    "OllamaGemmaProvider",
    "ScientificClaim",
    "StructuredEvidencePacket",
    "StructuredInterpretationResponse",
    "build_evidence_packet_from_utrnet_contrast",
    "evaluate_interpretation_grounding",
]
