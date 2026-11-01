# UTRNNet M1–M2 Experiment Report, Observed Metrics & Prioritized Backlog

## 1. Graph Construction Verification (`M1`)

Using the `cell-regulatory-state/v0.1` adapter ([`utrnnet.graphs.from_utrnet_directory`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py#L151-L810) and [`utrnnet.graphs.from_graphml`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py#L813-L1055)) and the upstream contract verifier ([`verify_upstream_to_cell_state_compatibility`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/utrnet_graph_v0_1.py#L244-L312)), frozen PBMC 1k (`pbmc_1k_v3`, 1,113 QC-passing cells across 16 curated cell types) outputs were converted and validated via [`validate_cell_regulatory_state`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/schema.py#L199-L325):

| Cohort / Gene | `population` | `gene` | `pas` | `utr_isoform` | `regulatory_region` | `regulator` | `cpa_factor` | Total Typed Edges | Schema Valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| **`SAT1`** (`ENSG00000130066`) | 16 | 1 | 26 | 34 | 2,577 | 95 | 37 | 15,066 | `True` |
| **`CORO1B`** (`ENSG00000172725`) | 16 | 1 | 10 | 17 | 4,333 | 192 | 36 | 24,654 | `True` |
| **Train Cohort (20 Genes Merged)** | 16 | 20 | 696 | 866 | 192,480 | 615 | 42 | 829,586 | `True` |
| **Held-Out Test Cohort (5 Disjoint Genes)** | 16 | 5 | 109 | 117 | 16,439 | 386 | 40 | 84,178 | `True` |

Key verified properties:
- All 4 epistemic edge categories (`structural`, `annotation_derived`, `observed`, `inferred_candidate`) are preserved in separate PyG edge stores.
- Endogenous miRNA regulators have `x_missing_mask[:, 0:2] == True` (`log1p_mean_umi_per_cell` and `abundance_factor` explicitly flagged as missing rather than imputed as zero).
- Unresolved and model-dependent 3'UTR isoforms (`ambiguity_code` and `is_unresolved_mask`) are explicitly retained on `utr_isoform` nodes and `("pas", "defines_isoform", "utr_isoform")` edges.
- Shared `population` and `regulator` nodes in multi-gene merged graphs mask gene-coupled summary features (`population.x[:, 2:5]` and `regulator.x[:, 5:7]`) so gene-specific APA and cis-dosage statistics never leak across genes.

---

## 2. Masked Node-Feature Reconstruction Benchmark (`M2` Consolidated)

All runs use `seed=42`, `hidden_channels=64`, `num_layers=2`, `heads=4`, `mask_ratio=0.25`, `lr=3e-3`, `log1p`-scaled cis-dosage features, and strict direct + indirect topological anti-leakage sanitization ([`apply_feature_mask_with_leakage_guard`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/leakage.py#L59-L265) + [`verify_zero_target_leakage`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/evaluation/metrics.py#L46-L108)).

### 2.1 Exploratory Two-Gene Panel (`SAT1` + `CORO1B`, 25 Epochs on CUDA)

| Model / Topology | Final Masked MSE | Final Masked MAE | Zero-Leakage Audit |
| --- | ---: | ---: | :---: |
| `TrivialNonGraphBaseline` (node-local MLP, no edges) | `0.0485` | `0.1615` | `True` |
| `UTRNNetHeteroGNN` + **Randomized-Edge Negative Control** | `0.0446` | `0.1510` | `True` |
| **`UTRNNetHeteroGNN` + True Regulatory Graph** | **`0.0330`** | **`0.1169`** | `True` |

### 2.2 Held-Out Inductive Gene Split (20 Train Genes $\rightarrow$ 5 Disjoint Unseen Test Genes) & Ablations

| Model / Ablation Variant | Held-Out Test MSE | Held-Out Test MAE | Relative MSE Change vs Baseline |
| --- | ---: | ---: | ---: |
| `TrivialNonGraphBaseline` (node-local MLP, no edges) | `0.0700` | `0.1797` | `0.0%` |
| `UTRNNetHeteroGNN` evaluated on **Randomized-Edge Negative Control** | `0.0756` | `0.2018` | `+8.0%` (degraded) |
| `UTRNNetHeteroGNN` (`structural + annotation_derived` only) | `0.0574` | `0.1601` | `-18.0%` |
| `UTRNNetHeteroGNN` (no `inferred_candidate` edges) | `0.0614` | `0.1619` | `-12.3%` |
| `UTRNNetHeteroGNN` (**Evidence Tiers Ablated** `use_evidence_tiers=False`) | `0.0602` | `0.1702` | `-14.0%` |
| **`UTRNNetHeteroGNN` (Full 4 Edge Categories + Evidence Tiers)** | **`0.0521`** | **`0.1428`** | **`-25.6%` (improved)** |

- **Resource footprint (25 genes + 6 models trained & evaluated):** `28.79 s` graph load, `63.88 s` total wall time, `4,741.62 MB` peak CUDA VRAM, `2,247.65 MB` peak CPU RSS.

> **Scientific interpretation note:** These metrics demonstrate self-supervised masked feature reconstruction and inductive generalization across held-out gene subgraphs on static observational PBMC1k data. They do **not** represent perturbation response prediction (evaluated separately in `docs/experiments/TAHOE_PILOT_BENCHMARK.md`) or validate physical RBP occupancy.

---

## 3. Unresolved Scientific Concerns

1. **Candidate sequence matches vs. physical occupancy:** ATtRACT RBP motif scans and MirGeneDB 7mer-m8 seed matches represent sequence-level priors (`inferred_candidate`), not cell-type-resolved physical binding. Direct eCLIP/CLIP-seq (e.g., ENCODE) integration is required to upgrade candidate edges to experimental occupancy tiers.
2. **Unmeasured endogenous miRNA abundance:** 10x 3' scPolASeq captures polyadenylated mRNAs and does not measure mature miRNA levels. All miRNA nodes remain prior-only (`x_missing_mask[:, 0:2] = True`).
3. **Transcript/PAS structural ambiguity:** Multiple GTF transcript models can share or split a Sierra PAS peak (`ambiguous_multiple_models`), and some intronic/proximal PAS peaks lack a compatible annotated terminal-exon 3'UTR (`no_compatible_3utr_model`).
4. **Cohort size and single-donor PBMC 1k scope:** PBMC 1k is a single biological sampling context without donor replicates.
5. **No continuous-time dynamics on static cell types:** Curated PBMC cell types are not temporal observations (`is_temporal_observation = False`).
