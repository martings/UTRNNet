# UTRNNet M1–M2 Experiment Report, Observed Metrics & Prioritized Backlog

## 1. Graph Construction Verification (`M1`)

Using the `cell-regulatory-state/v0.1` adapter (`utrnnet.graphs.from_utrnet_directory` and `utrnnet.graphs.from_graphml`), frozen PBMC 1k (`pbmc_1k_v3`, 1,113 QC-passing cells across 16 curated cell types) outputs were converted and validated via `validate_cell_regulatory_state`:

| Cohort / Gene | `population` | `gene` | `pas` | `utr_isoform` | `regulatory_region` | `regulator` | `cpa_factor` | Total Typed Edges | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **`SAT1`** (`ENSG00000130066`) | 16 | 1 | 26 | 34 | 2,577 | 95 | 37 | 15,066 | `True` |
| **`CORO1B`** (`ENSG00000172725`) | 16 | 1 | 10 | 17 | 4,333 | 192 | 36 | 24,654 | `True` |
| **Merged 25-Gene PBMC1k Cohort** | 16 | 25 | 805 | 983 | 208,919 | 618 | 42 | 901,508 | `True` |

Key verified properties:
- All 4 epistemic edge categories (`structural`, `annotation_derived`, `observed`, `inferred_candidate`) are preserved in separate PyG edge stores.
- Endogenous miRNA regulators have `x_missing_mask[:, 0:2] == True` (`log1p_mean_umi_per_cell` and `abundance_factor` explicitly flagged as missing rather than imputed as zero).
- Unresolved and model-dependent 3'UTR isoforms (`ambiguity_code` and `is_unresolved_mask`) are explicitly retained on `utr_isoform` nodes and `("pas", "defines_isoform", "utr_isoform")` edges.

---

## 2. Masked Node-Feature Reconstruction Benchmark (`M2`)

All runs use `seed=42`, `hidden_channels=64`, `num_layers=2`, `heads=4`, `mask_ratio=0.25`, `lr=3e-3`, and strict topological anti-leakage sanitization (`apply_feature_mask_with_leakage_guard` + `verify_zero_target_leakage`).

### 2.1 Exploratory Two-Gene Panel (`SAT1` + `CORO1B`, 25 Epochs on CUDA)

| Model / Topology | Final Masked MSE | Final Masked MAE | Zero-Leakage Audit |
| --- | ---: | ---: | :---: |
| `TrivialNonGraphBaseline` (node-local MLP, no edges) | `4.7585` | `0.2653` | `True` |
| `UTRNNetHeteroGNN` + **Randomized-Edge Negative Control** | `4.7092` | `0.2500` | `True` |
| `UTRNNetHeteroGNN` + **True Regulatory Graph** | **`4.6716`** | **`0.2064`** | `True` |

### 2.2 Multi-Gene PBMC1k Cohort (25 QC-Passing APA Genes, 901,508 Typed Edges)

#### Transductive Masked Feature Reconstruction (All 25 Genes Merged)

| Model / Topology | Final Masked MSE | Final Masked MAE | Zero-Leakage Audit |
| --- | ---: | ---: | :---: |
| `TrivialNonGraphBaseline` (node-local MLP, no edges) | `0.3908` | `0.1683` | `True` |
| `UTRNNetHeteroGNN` + **Randomized-Edge Negative Control** | `0.3886` | `0.1598` | `True` |
| `UTRNNetHeteroGNN` + **True Regulatory Graph** | **`0.3719`** | **`0.1353`** | `True` |

#### Held-Out Inductive Gene Split (20 Train Genes $\rightarrow$ 5 Unseen Test Genes)

| Model / Test Topology | Held-Out Test MSE | Held-Out Test MAE | Relative MAE Change vs Baseline |
| --- | ---: | ---: | ---: |
| `TrivialNonGraphBaseline` (node-local MLP, no edges) | `0.7190` | `0.2117` | `0.0%` |
| `UTRNNetHeteroGNN` evaluated on **Randomized-Edge Negative Control** | `0.7862` | `0.2521` | `+19.1%` (degraded) |
| `UTRNNetHeteroGNN` evaluated on **True Regulatory Graph** | **`0.6564`** | **`0.1634`** | **`-22.8%` (improved)** |

> **Scientific interpretation note:** These metrics demonstrate only that typed message passing over structural/annotation/observed/candidate graph topology improves held-out masked feature reconstruction relative to an edge-free MLP and a randomized-edge topology without target leakage. They do **not** validate physical RBP occupancy or causal regulatory dynamics.

---

## 3. Unresolved Scientific Concerns

1. **Candidate sequence matches vs. physical occupancy:** ATtRACT RBP motif scans and MirGeneDB 7mer-m8 seed matches represent sequence-level priors (`inferred_candidate`), not cell-type-resolved physical binding. Direct eCLIP/CLIP-seq (e.g., ENCODE) integration is required to upgrade candidate edges to experimental occupancy tiers.
2. **Unmeasured endogenous miRNA abundance:** 10x 3' scPolASeq captures polyadenylated mRNAs and does not measure mature miRNA levels. All miRNA nodes remain prior-only (`x_missing_mask[:, 0:2] = True`).
3. **Transcript/PAS structural ambiguity:** Multiple GTF transcript models can share or split a Sierra PAS peak (`ambiguous_multiple_models`), and some intronic/proximal PAS peaks lack a compatible annotated terminal-exon 3'UTR (`no_compatible_3utr_model`).
4. **Cohort size and single-donor PBMC 1k scope:** While 25 pre-built PBMC 1k APA genes (901,508 edges) were evaluated here and the loader supports 30–100+ genes, PBMC 1k is a single biological sampling context without donor replicates.
5. **No continuous-time dynamics on static cell types:** Curated PBMC cell types are not temporal observations (`is_temporal_observation = False`). Longitudinal or perturbation modeling must wait for multi-timepoint / CRISPRi / RBP-knockdown datasets.

---

## 4. Prioritized Issue Backlog

- **Issue #1 (P0 — Next Recommended): Expand PBMC1k & Multi-Donor Cohort Cache to 50–100 Quality-Controlled APA Genes with Pre-Indexed GTF/FASTA Streaming**
  - Pre-index GTF/Sierra tables once per cohort so `scpolaseq_utrnet.pipeline.build_gene_graphs` builds 100+ QC-passing APA genes in `< 30s`, and evaluate 5-fold cross-validation across gene families.
- **Issue #2 (P1): Edge-Category Ablation & Relation-Specific Attention Attribution**
  - Quantify the marginal reconstruction contribution of `structural` vs `annotation_derived` vs `observed` vs `inferred_candidate` edges by training `UTRNNetHeteroGNN` across subsets of `allowed_edge_categories`.
- **Issue #3 (P1): ENCODE eCLIP / CLIP-seq Experimental Occupancy Tier Integration**
  - Cross-reference `regulatory_region` genomic intervals against ENCODE eCLIP peaks (K562/HepG2/immune) to populate `experimental_other_context` and `experimental_matching_context` evidence tiers with measured peak scores.
- **Issue #4 (P2): Out-of-Fold PAS Usage & Cis-Dosage Prediction Task (Strict Expression-Free Input Mode)**
  - Train a dedicated head predicting population-specific PAS usage proportions (`pas_proportion`) from CPA factor expression, RBP abundance, and sequence/accessibility priors alone (`strip_expression_from_candidate_edges=True`).
- **Issue #5 (P2): Validate `PerturbationConditionedHead` on Experimental RBP Knockdown / CRISPRi Perturb-seq Cohorts**
  - Benchmark `UTRNNetHeteroGNN.predict_perturbation_delta` on experimental RBP knockdown datasets (e.g., ENCODE shRNA RNA-seq or Perturb-seq) before making any causal perturbation claims.
- **Issue #6 (P3): Discrete Paired Longitudinal State-Transition Adapter (Dengue Day 1 $\rightarrow$ Week 2)**
  - Extend `cell-regulatory-state/v0.1` to paired longitudinal cohorts (`is_temporal_observation = True`) using discrete paired transitions before considering any continuous-time neural ODE formulation.
