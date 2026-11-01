"""Minimum paired-condition data contract (`dengue-paired-longitudinal/v0.1`) for Dengue day-1 vs week-2 experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DENGUE_CONTRACT_VERSION = "dengue-paired-longitudinal/v0.1"

REQUIRED_SAMPLE_METADATA_FIELDS: tuple[str, ...] = (
    "sample_id",
    "patient_id",
    "condition_label",
    "timepoint_label",
    "timepoint_order",
    "clinical_severity",
    "celltype_label",
    "n_cells",
)

CANONICAL_DENGUE_CONDITIONS: dict[str, dict[str, Any]] = {
    "DF": {"clinical_severity": "DF", "timepoint_label": "day_1_acute", "timepoint_order": 0},
    "DF_wk2": {"clinical_severity": "DF", "timepoint_label": "week_2_convalescent", "timepoint_order": 1},
    "DHF": {"clinical_severity": "DHF", "timepoint_label": "day_1_acute", "timepoint_order": 0},
    "DHF_wk2": {"clinical_severity": "DHF", "timepoint_label": "week_2_convalescent", "timepoint_order": 1},
}


@dataclass(frozen=True)
class DengueSampleStateRecord:
    """Single population-level state observation in a discrete longitudinal Dengue cohort."""

    sample_id: str
    patient_id: str
    condition_label: str
    timepoint_label: str
    timepoint_order: int
    clinical_severity: str
    celltype_label: str
    n_cells: int
    gene_id: str
    has_matched_single_cells: bool = False


@dataclass(frozen=True)
class DengueContrastPairSpec:
    """Explicit specification of a discrete state contrast (`t0 -> t1` or between-group exploratory)."""

    contrast_id: str
    source_sample: DengueSampleStateRecord
    target_sample: DengueSampleStateRecord
    is_within_patient_longitudinal: bool
    is_within_severity_group: bool
    is_exploratory_between_subject_contrast: bool
    has_matched_single_cells: bool = False
    continuous_time_ode_allowed: bool = False


@dataclass(frozen=True)
class DengueContractValidationReport:
    """Validation result for a `dengue-paired-longitudinal/v0.1` cohort and contrast set."""

    valid: bool
    contract_version: str
    n_samples: int
    n_within_patient_pairs: int
    n_exploratory_contrasts: int
    continuous_time_ode_allowed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_and_validate_dengue_contrast(
    source: DengueSampleStateRecord,
    target: DengueSampleStateRecord,
    *,
    allow_matched_single_cells: bool = False,
    allow_continuous_time_ode: bool = False,
) -> DengueContrastPairSpec:
    """Construct a validated discrete contrast pair while enforcing longitudinal safety guards."""
    if source.celltype_label != target.celltype_label:
        raise ValueError(
            f"Cannot pair mismatched cell populations: {source.celltype_label!r} vs {target.celltype_label!r}"
        )
    if source.gene_id != target.gene_id:
        raise ValueError(f"Cannot pair mismatched genes: {source.gene_id!r} vs {target.gene_id!r}")
    if source.has_matched_single_cells or target.has_matched_single_cells or allow_matched_single_cells:
        raise ValueError(
            "Destructive scPolASeq droplets cannot be treated as matched longitudinal single cells "
            "(has_matched_single_cells must be False)."
        )
    if allow_continuous_time_ode:
        raise ValueError(
            "Continuous-time neural ODEs are prohibited on 2-timepoint (day-1 vs week-2) Dengue cohorts; "
            "use discrete paired population-state transitions."
        )

    same_patient = (
        bool(source.patient_id)
        and bool(target.patient_id)
        and source.patient_id != "unpaired_pool"
        and source.patient_id == target.patient_id
    )
    same_severity = source.clinical_severity == target.clinical_severity
    forward_in_time = source.timepoint_order < target.timepoint_order

    is_within_patient_longitudinal = same_patient and same_severity and forward_in_time
    is_exploratory = not is_within_patient_longitudinal

    cid = (
        f"{source.gene_id}:{source.celltype_label}:"
        f"{source.condition_label}({source.patient_id})->{target.condition_label}({target.patient_id})"
    )
    return DengueContrastPairSpec(
        contrast_id=cid,
        source_sample=source,
        target_sample=target,
        is_within_patient_longitudinal=is_within_patient_longitudinal,
        is_within_severity_group=same_severity,
        is_exploratory_between_subject_contrast=is_exploratory,
        has_matched_single_cells=False,
        continuous_time_ode_allowed=False,
    )


def validate_dengue_longitudinal_cohort(
    samples: list[DengueSampleStateRecord],
    contrasts: list[DengueContrastPairSpec],
    *,
    min_cells_per_population: int = 15,
    raise_on_error: bool = True,
) -> DengueContractValidationReport:
    """Validate a collection of Dengue sample records and discrete contrast pairs against `dengue-paired-longitudinal/v0.1`."""
    errors: list[str] = []
    warnings: list[str] = []

    if not samples:
        errors.append("Cohort contains zero sample state records")

    for s in samples:
        if not s.sample_id or not s.patient_id:
            errors.append(f"Sample {s} is missing explicit sample_id or patient_id")
        if s.condition_label not in CANONICAL_DENGUE_CONDITIONS:
            warnings.append(f"Non-canonical condition_label {s.condition_label!r} on sample {s.sample_id!r}")
        if s.has_matched_single_cells:
            errors.append(
                f"Sample {s.sample_id!r} sets has_matched_single_cells=True; scRNA-seq cells are not longitudinal pairs"
            )
        if s.n_cells < min_cells_per_population:
            warnings.append(
                f"Sample {s.sample_id!r} ({s.celltype_label}) has n_cells={s.n_cells} < {min_cells_per_population}"
            )

    n_within_patient = 0
    n_exploratory = 0
    for c in contrasts:
        if c.has_matched_single_cells:
            errors.append(f"Contrast {c.contrast_id!r} illegally claims matched single cells")
        if c.continuous_time_ode_allowed:
            errors.append(f"Contrast {c.contrast_id!r} illegally enables continuous_time_ode_allowed on 2 timepoints")
        if c.source_sample.patient_id != c.target_sample.patient_id and not c.is_exploratory_between_subject_contrast:
            errors.append(
                f"Cross-patient contrast {c.contrast_id!r} must be flagged is_exploratory_between_subject_contrast=True"
            )
        if (
            c.source_sample.clinical_severity != c.target_sample.clinical_severity
            and not c.is_exploratory_between_subject_contrast
        ):
            errors.append(
                f"Cross-severity contrast {c.contrast_id!r} (DF vs DHF) must be flagged exploratory"
            )
        if c.is_within_patient_longitudinal:
            n_within_patient += 1
        if c.is_exploratory_between_subject_contrast:
            n_exploratory += 1

    report = DengueContractValidationReport(
        valid=len(errors) == 0,
        contract_version=DENGUE_CONTRACT_VERSION,
        n_samples=len(samples),
        n_within_patient_pairs=n_within_patient,
        n_exploratory_contrasts=n_exploratory,
        continuous_time_ode_allowed=False,
        errors=errors,
        warnings=warnings,
    )
    if errors and raise_on_error:
        raise ValueError("Invalid dengue-paired-longitudinal/v0.1 cohort:\n- " + "\n- ".join(errors))
    return report
