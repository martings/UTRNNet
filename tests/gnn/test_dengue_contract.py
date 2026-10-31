"""Task E tests: Minimum paired-condition data contract (`dengue-paired-longitudinal/v0.1`)."""
from __future__ import annotations

import pytest

from utrnnet.contracts import (
    DENGUE_CONTRACT_VERSION,
    DengueSampleStateRecord,
    build_and_validate_dengue_contrast,
    validate_dengue_longitudinal_cohort,
)


def test_dengue_paired_within_patient_vs_exploratory_contrasts():
    s_df_d1_p1 = DengueSampleStateRecord(
        sample_id="smp_DF_P01_d1",
        patient_id="P01",
        condition_label="DF",
        timepoint_label="day_1_acute",
        timepoint_order=0,
        clinical_severity="DF",
        celltype_label="Classical monocytes",
        n_cells=120,
        gene_id="ENSG00000130066",
        has_matched_single_cells=False,
    )
    s_df_wk2_p1 = DengueSampleStateRecord(
        sample_id="smp_DF_P01_wk2",
        patient_id="P01",
        condition_label="DF_wk2",
        timepoint_label="week_2_convalescent",
        timepoint_order=1,
        clinical_severity="DF",
        celltype_label="Classical monocytes",
        n_cells=95,
        gene_id="ENSG00000130066",
        has_matched_single_cells=False,
    )
    s_dhf_d1_p2 = DengueSampleStateRecord(
        sample_id="smp_DHF_P02_d1",
        patient_id="P02",
        condition_label="DHF",
        timepoint_label="day_1_acute",
        timepoint_order=0,
        clinical_severity="DHF",
        celltype_label="Classical monocytes",
        n_cells=88,
        gene_id="ENSG00000130066",
        has_matched_single_cells=False,
    )

    # 1. Within-patient day-1 -> week-2 contrast
    c_paired = build_and_validate_dengue_contrast(s_df_d1_p1, s_df_wk2_p1)
    assert c_paired.is_within_patient_longitudinal is True
    assert c_paired.is_within_severity_group is True
    assert c_paired.is_exploratory_between_subject_contrast is False
    assert c_paired.has_matched_single_cells is False
    assert c_paired.continuous_time_ode_allowed is False

    # 2. Cross-patient / cross-severity contrast (DF vs DHF) MUST be flagged exploratory
    c_cross = build_and_validate_dengue_contrast(s_df_d1_p1, s_dhf_d1_p2)
    assert c_cross.is_within_patient_longitudinal is False
    assert c_cross.is_within_severity_group is False
    assert c_cross.is_exploratory_between_subject_contrast is True

    rep = validate_dengue_longitudinal_cohort(
        [s_df_d1_p1, s_df_wk2_p1, s_dhf_d1_p2],
        [c_paired, c_cross],
        raise_on_error=True,
    )
    assert rep.valid
    assert rep.contract_version == DENGUE_CONTRACT_VERSION
    assert rep.n_within_patient_pairs == 1
    assert rep.n_exploratory_contrasts == 1
    assert rep.continuous_time_ode_allowed is False


def test_dengue_contract_rejects_matched_single_cells_and_continuous_time_ode():
    s1 = DengueSampleStateRecord(
        sample_id="s1",
        patient_id="P01",
        condition_label="DF",
        timepoint_label="day_1_acute",
        timepoint_order=0,
        clinical_severity="DF",
        celltype_label="Monocytes",
        n_cells=50,
        gene_id="ENSG00000130066",
    )
    s2 = DengueSampleStateRecord(
        sample_id="s2",
        patient_id="P01",
        condition_label="DF_wk2",
        timepoint_label="week_2_convalescent",
        timepoint_order=1,
        clinical_severity="DF",
        celltype_label="Monocytes",
        n_cells=50,
        gene_id="ENSG00000130066",
    )

    with pytest.raises(ValueError, match="matched longitudinal single cells"):
        build_and_validate_dengue_contrast(s1, s2, allow_matched_single_cells=True)

    with pytest.raises(ValueError, match="Continuous-time neural ODEs are prohibited"):
        build_and_validate_dengue_contrast(s1, s2, allow_continuous_time_ode=True)
