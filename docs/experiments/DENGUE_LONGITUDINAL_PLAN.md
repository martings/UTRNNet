# Dengue Longitudinal Experiment Plan & Paired-Condition Contract (`Task E`)

## 1. Scientific Scope & Guardrails

The Dengue scPolASeq cohort captures peripheral blood immune cells across clinical severity groups (`DF` = Dengue Fever, `DHF` = Dengue Hemorrhagic Fever) and two discrete clinical timepoints (`day_1_acute` vs `week_2_convalescent`, encoded as `DF`, `DF_wk2`, `DHF`, `DHF_wk2`).

To prevent methodological invalidation, [`src/utrnnet/contracts/dengue_longitudinal.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/dengue_longitudinal.py) (`dengue-paired-longitudinal/v0.1`) enforces four hard scientific invariants:

1. **Preservation of Patient, Sample, and Time Identifiers:** Every population state record ([`DengueSampleStateRecord`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/dengue_longitudinal.py#L26-L38)) must carry explicit `sample_id`, `patient_id`, `condition_label`, `timepoint_label`, `timepoint_order`, `clinical_severity`, `celltype_label`, and `n_cells`.
2. **Single Cells Are Never Matched Longitudinal Observations:** Because droplet 3' scPolASeq is destructive, cells at `day_1_acute` and `week_2_convalescent` are distinct cells sampled from the same patient (`has_matched_single_cells = False`). Any attempt to set `has_matched_single_cells = True` raises a `ValueError`.
3. **Within-Patient Paired Transitions vs. Exploratory Between-Subject Contrasts:**
   - Only contrasts with `source.patient_id == target.patient_id`, `source.clinical_severity == target.clinical_severity`, `source.celltype_label == target.celltype_label`, and `source.timepoint_order < target.timepoint_order` (`day_1_acute -> week_2_convalescent`) qualify as `is_within_patient_longitudinal = True`.
   - Any comparison across different patients (`patient_id_a != patient_id_b`) or across severity groups (`DF` vs `DHF`) is automatically flagged `is_exploratory_between_subject_contrast = True`.
4. **No Continuous-Time Neural ODEs on Two Timepoints:** With only two discrete clinical timepoints (`t0 = day_1_acute`, `t1 = week_2_convalescent`), continuous-time neural ODE trajectories are unidentifiable (`continuous_time_ode_allowed = False`). All longitudinal modeling is restricted to discrete paired population-state transition operators $\hat{S}(t_1) = S(t_0) + f_\theta(G(t_0), \Delta_{\text{regulator}}(t_0 \to t_1))$.

---

## 2. Minimum Paired-Condition Data Contract (`dengue-paired-longitudinal/v0.1`)

| Field | Type | Required | Constraint / Meaning |
| --- | --- | :---: | --- |
| `sample_id` | `str` | Yes | Unique sequencing library / sample identifier |
| `patient_id` | `str` | Yes | Donor / subject identifier (never `"unpaired_pool"` for longitudinal claims) |
| `condition_label` | `str` | Yes | Canonical group (`DF`, `DF_wk2`, `DHF`, `DHF_wk2`) |
| `timepoint_label` | `str` | Yes | `day_1_acute` (`order=0`) or `week_2_convalescent` (`order=1`) |
| `clinical_severity` | `str` | Yes | `DF` or `DHF` |
| `celltype_label` | `str` | Yes | Matched CellTypist immune lineage (e.g., `Classical monocytes`, `Plasmablasts`) |
| `n_cells` | `int` | Yes | Minimum `>= 15` QC-passing cells per `(patient_id, timepoint, celltype)` |
| `has_matched_single_cells` | `bool` | Yes | Must be `False` (destructive assay guard) |
| `continuous_time_ode_allowed` | `bool` | Yes | Must be `False` on 2-timepoint discrete cohorts |

---

## 3. Staged Experimental Design for Dengue Day-1 $\rightarrow$ Week-2

1. **Stage E1 — Metadata & Donor-Pairing Audit:**
   - Audit `dengue_harmonized_annotated.h5ad` and Sierra PAS count matrices to enumerate all `(patient_id, celltype_label)` pairs with `n_cells >= 15` at both `day_1_acute` and `week_2_convalescent`.
2. **Stage E2 — Discrete Paired Population-Graph Construction:**
   - Build per-patient, per-timepoint `cell-regulatory-state/v0.1` graphs for top APA-shifting immune genes (`SAT1`, `CORO1B`, `IFITM1`, `ISG20`, `IGHM`) and validate with [`validate_dengue_longitudinal_cohort`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/dengue_longitudinal.py#L96-L157).
3. **Stage E3 — Discrete State-Transition Benchmark (Leave-One-Patient-Out):**
   - Compare (1) zero-transition baseline ($\hat{\Delta}_{t_0 \to t_1} = 0$), (2) Ridge baseline on $t_0$ gene + CPA/RBP expression, and (3) discrete paired `UTRNNetHeteroGNN` transition head across held-out patients (`validate_group_partition(group_column="patient_id")`).
   - Keep `DF` vs `DHF` severity contrasts in a separate exploratory report (`is_exploratory_between_subject_contrast = True`).
