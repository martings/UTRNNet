"""UTRNNet: UTR Regulatory Neural Network extension of scPolASeq-UTRNet."""
from __future__ import annotations

from utrnnet import contracts, data, evaluation, graphs, interpretation, models, tahoe, training
from utrnnet.contracts import (
    CANONICAL_EDGE_TYPES,
    DENGUE_CONTRACT_VERSION,
    EDGE_CATEGORIES,
    NODE_FEATURE_SCHEMAS,
    NODE_TYPES,
    SCHEMA_VERSION,
    UPSTREAM_SCHEMA_VERSION,
    ValidationReport,
    validate_cell_regulatory_state,
    validate_dengue_longitudinal_cohort,
    validate_utrnet_graphml,
    validate_utrnet_upstream_directory,
    verify_upstream_to_cell_state_compatibility,
)
from utrnnet.data import load_gene_cohort_graphs
from utrnnet.evaluation import (
    BenchmarkComparison,
    evaluate_masked_reconstruction,
    run_three_way_benchmark,
    verify_zero_target_leakage,
)
from utrnnet.graphs import (
    MaskedGraphBatch,
    apply_feature_mask_with_leakage_guard,
    from_graphml,
    from_utrnet_directory,
    load_hetero_data,
    merge_hetero_graphs,
    randomize_hetero_graph_edges,
    save_hetero_data,
    strip_target_expression_leakage,
)
from utrnnet.models import (
    PerturbationConditionedHead,
    TrivialNonGraphBaseline,
    TypedHeteroMessagePassingLayer,
    UTRNNetHeteroGNN,
)
from utrnnet.training import (
    TrainingConfig,
    TrainingResult,
    compute_masked_reconstruction_loss,
    resolve_device,
    set_reproducible_seed,
    train_masked_reconstruction,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "BenchmarkComparison",
    "CANONICAL_EDGE_TYPES",
    "DENGUE_CONTRACT_VERSION",
    "EDGE_CATEGORIES",
    "MaskedGraphBatch",
    "NODE_FEATURE_SCHEMAS",
    "NODE_TYPES",
    "PerturbationConditionedHead",
    "SCHEMA_VERSION",
    "TrainingConfig",
    "TrainingResult",
    "TrivialNonGraphBaseline",
    "TypedHeteroMessagePassingLayer",
    "UPSTREAM_SCHEMA_VERSION",
    "UTRNNetHeteroGNN",
    "ValidationReport",
    "apply_feature_mask_with_leakage_guard",
    "compute_masked_reconstruction_loss",
    "contracts",
    "data",
    "evaluate_masked_reconstruction",
    "evaluation",
    "from_graphml",
    "from_utrnet_directory",
    "graphs",
    "interpretation",
    "load_gene_cohort_graphs",
    "load_hetero_data",
    "merge_hetero_graphs",
    "models",
    "randomize_hetero_graph_edges",
    "resolve_device",
    "run_three_way_benchmark",
    "save_hetero_data",
    "set_reproducible_seed",
    "strip_target_expression_leakage",
    "tahoe",
    "train_masked_reconstruction",
    "training",
    "validate_cell_regulatory_state",
    "validate_dengue_longitudinal_cohort",
    "validate_utrnet_graphml",
    "validate_utrnet_upstream_directory",
    "verify_upstream_to_cell_state_compatibility",
    "verify_zero_target_leakage",
]
