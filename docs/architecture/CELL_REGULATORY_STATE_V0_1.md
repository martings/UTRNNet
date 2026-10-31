# UTRNNet Architecture and `cell-regulatory-state/v0.1` Data Contract

## 1. Repository Provenance & Initialization (`M0`)

- **Upstream source repository:** `https://github.com/martings/scpolaseq_utrnet` (configured as a read-only remote `upstream` with `pushurl = no_push://read-only-upstream`).
- **Upstream base branch & commit SHA:** `main` at `02997cb5a5e170200d1efeb42880050e3de8cfdd` (`feat(viz): add explanatory cell-type regulatory network and contrast visualizer`).
- **Target repository:** `https://github.com/martings/UTRNNet` initialized with full Git commit history and `v0.2.0` tag preserved from `scpolaseq_utrnet`.
- **Backward compatibility guarantee:** The original `scpolaseq_utrnet` package and `utrnet` CLI remain untouched under `src/scpolaseq_utrnet/`, while the PyTorch Geometric extension lives in an independent sibling package `src/utrnnet/` (`utrnnet` CLI).

---

## 2. Typed Heterogeneous Graph Schema (`cell-regulatory-state/v0.1`)

Every converted `torch_geometric.data.HeteroData` graph carries:
- `schema_version = "cell-regulatory-state/v0.1"`
- `is_temporal_observation = False` by default (preventing static cell types from being misinterpreted as longitudinal trajectories).

### 2.1 Node Types and Feature Contracts

Every node store `data[ntype]` defines:
- `x`: `float32` tensor of shape `[N, D]` with neutral numeric fill (`0.0`) at missing positions.
- `x_missing_mask`: `bool` tensor of shape `[N, D]` (`True` where the raw value was `NaN` or unmeasured).
- `feature_names`: ordered tuple of feature names.
- `node_ids`: deterministic list of canonical string identifiers (e.g., `gene:ENSG00000130066`, `pas:...`, `utr:...`, `region:...`, `regulator:ELAVL1`).
- `node_id_to_index`: reversible `dict[str, int]` mapping.

| Node Type | Features (`D`) | Scientific & Missingness Rules |
| --- | --- | --- |
| `population` | `log1p_n_cells`, `meets_min_cells`, `mean_shortening_index`, `mean_magnetization_genomic`, `mean_apa_entropy` (`5`) | Represents a curated cell population/state, **not** a time point (`is_temporal_observation = False`). |
| `gene` | `log1p_mean_gene_umi`, `n_pas_sites`, `n_utr_isoforms`, `shortening_index`, `magnetization_genomic`, `apa_entropy` (`6`) | Target gene node; when predicting gene expression, incident `expresses_gene` edges are sanitized. |
| `pas` | `relative_coord_norm`, `strand_sign`, `pas_proportion`, `log1p_pas_umi`, `usage_ci_width` (`5`) | Cleavage site from Sierra / PAS reference. Zero-coverage PAS proportions are flagged in `x_missing_mask`. |
| `utr_isoform` | `log1p_sequence_length`, `model_weight`, `pas_proportion`, `is_unique_model`, `is_model_dependent`, `is_unresolved_model`, `is_suspicious_span` (`7`) | Explicit `ambiguity_code` (`0=unique`, `1=ambiguous_multiple_models`, `2=no_compatible_3utr_model`) and `is_unresolved_mask`. |
| `regulatory_region` | `relative_region_pos`, `region_length_bp`, `accessibility`, `survival_probability`, `is_autr_region`, `evidence_factor` (`6`) | Uncomputed `RNAplfold` accessibility is flagged in `x_missing_mask`. Includes continuous motif survival $S(m; c)$ from `scpolaseq_utrnet.cis`. |
| `regulator` | `log1p_mean_umi_per_cell`, `abundance_factor`, `is_rbp`, `is_mirna`, `max_evidence_factor`, `fractional_cis_dosage_total`, `fractional_cis_dosage_autr` (`7`) | **Endogenous miRNA abundance is unmeasured in 10x 3' scPolASeq**; `x_missing_mask[:, 0:2]` MUST be `True` for all miRNA nodes. |
| `cpa_factor` | `log1p_mean_expr`, `pct_expressing` (`2`) | Optional 3' end cleavage & polyadenylation factor nodes when CPA tables are present. |

### 2.2 Epistemic Separation of Edge Categories

Structural, annotation-derived, observed, and inferred candidate edges are stored as distinct relation types with explicit `edge_category`, `evidence_tier_id`, `edge_attr`, `edge_attr_missing_mask`, and `is_candidate_only`:

1. **`structural`** (`is_candidate_only = False`):
   - `("gene", "has_pas", "pas")` and `("pas", "rev_has_pas", "gene")` (`evidence="observed_reference"`)
   - `("utr_isoform", "contains_site", "regulatory_region")` and `("regulatory_region", "rev_contains_site", "utr_isoform")` (`evidence="sequence_or_import"`)
2. **`annotation_derived`** (`is_candidate_only = False`):
   - `("pas", "defines_isoform", "utr_isoform")` and `("utr_isoform", "rev_defines_isoform", "pas")` (`evidence="annotation_model"`, carrying `model_weight`, `is_ambiguous_model`, `is_unresolved_model`)
3. **`observed`** (`is_candidate_only = False`):
   - `("population", "uses_isoform", "utr_isoform")` and reverse (`evidence="Sierra_UMI"`)
   - `("population", "expresses_gene", "gene")` and reverse (`evidence="curated_UMI"`, `uses_target_expression = True`)
   - `("population", "expresses_regulator", "regulator")` and reverse (`evidence="curated_UMI"`)
4. **`inferred_candidate`** (`is_candidate_only = True`):
   - `("regulatory_region", "candidate_binding", "regulator")` and reverse (`evidence=evidence_tier`)
   - `("regulator", "candidate_regulation", "gene")` and reverse (`evidence=evidence_tier`)
   - `("cpa_factor", "modulates_apa", "gene")` and reverse (`evidence=evidence_tier`)

---

## 3. Anti-Leakage Guarantees (`utrnnet.graphs.leakage`)

Because UTRNet `interaction_score` incorporates `relative_isoform_usage` and `abundance_factor`, and `cerna` `gene_load` incorporates `gene_expression`:
1. `cerna` `gene_load` and `cerna_crosstalk` edges are excluded from target-gene prediction inputs.
2. During masked node-feature reconstruction (`apply_feature_mask_with_leakage_guard`), `interaction_score` (column 3 on `inferred_candidate` edges) is always zeroed and marked missing (`edge_attr_missing_mask[:, 3] = True`), leaving only sequence/cis-dosage, accessibility, and evidence-tier priors on candidate edges.
3. Topological cross-masking automatically sanitizes:
   - `("population", "expresses_gene", "gene")` edge attributes when `gene` expression (`col 0`) is masked.
   - `("population", "uses_isoform", "utr_isoform")` edge attributes and coupled `pas` $\leftrightarrow$ `utr_isoform` usage features when PAS/isoform usage is masked.
   - `("population", "expresses_regulator", "regulator")` edge attributes and coupled `abundance_factor` when regulator mRNA abundance is masked.
4. `verify_zero_target_leakage` mathematically verifies $\max |\hat{y}(\text{base}) - \hat{y}(\text{corrupted target}+500)| = 0.0$.

---

## 4. Model Architecture (`M2`)

- **`TrivialNonGraphBaseline`**: Node-local 2-layer MLP taking `[x * (~mask), (~mask)] -> 64 -> D_out`, ignoring all graph edges.
- **`UTRNNetHeteroGNN`**:
  - Missingness-aware node input projections `[x * (~mask), (~mask)] -> 64`.
  - Two `TypedHeteroMessagePassingLayer` blocks using multi-head `TransformerConv` (`hidden_channels=64, heads=4`) conditioned on missingness-masked edge attributes and learned `evidence_tier_id` embeddings.
  - Per-node-type reconstruction decoders (`64 -> 64 -> D_out`) trained with masked MSE/MAE over observed features only.
  - `PerturbationConditionedHead`: Discrete perturbation-conditioning scaffold for predicting post-perturbation $\Delta \text{gene}$ and $\Delta \text{utr\_isoform}$ states from $\Delta \text{regulator}$ interventions (no continuous-time ODE assumptions).
