# Technical Review & Consolidation Report for UTRNNet PR #1 (`Task A`)

- **Target PR:** [`martings/UTRNNet#1`](https://github.com/martings/UTRNNet/pull/1) (`feat/m0-m2-utrnnet-hetero-gnn`)
- **Upstream Baseline:** [`martings/scpolaseq_utrnet@02997cb`](https://github.com/martings/scpolaseq_utrnet/commit/02997cb5a5e170200d1efeb42880050e3de8cfdd) (preserved unmodified; configured as read-only remote)

---

## 1. Ten-Point Verification Audit

### 1.1 Reproducibility of Reported Metrics
- Deterministic seeding via [`set_reproducible_seed`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/training/trainer.py#L22-L31) fixes `random`, `numpy`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`, while [`apply_feature_mask_with_leakage_guard`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/leakage.py#L59-L265) uses an isolated CPU `torch.Generator` seeded with `seed=42`.
- All reported metrics on both the 2-gene (`SAT1` + `CORO1B`) and 25-gene PBMC1k cohorts are reproducible across runs.

### 1.2 Train/Test Graph Construction & 1.3 Gene-Level Split Isolation
- **Pre-fix issue identified:** In the initial PR #1 script, all 25 genes were merged via [`merge_hetero_graphs`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py#L1058-L1208) prior to evaluation, which averaged gene-specific APA summary metrics (`population.x[:, 2:5]`) and fractional cis-dosages (`regulator.x[:, 5:7]`) across all 25 genes onto shared `population` and `regulator` nodes.
- **Consolidation fix applied:**
  1. In [`merge_hetero_graphs`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py#L1058-L1208), when merging multiple distinct genes (`is_multi_gene=True`, `strip_gene_coupled_shared_node_features=True`), gene-coupled columns on shared `population` (`cols 2:5`) and `regulator` (`cols 5:7`) nodes are explicitly masked (`0.0`, `x_missing_mask=True`) so shared nodes retain only intrinsic population/regulator attributes (`log1p_n_cells`, `meets_min_cells`, `log1p_mean_umi_per_cell`, `abundance_factor`, `is_rbp`, `is_mirna`, `max_evidence_factor`), while gene-coupled quantities stay strictly on `gene` nodes and `("regulator", "candidate_regulation", "gene")` edges.
  2. In inductive evaluation, `train_graph` (20 genes, `829,586` edges) and `test_graph` (5 held-out genes, `84,178` edges) are constructed and merged **independently**, guaranteeing zero overlap of `gene`, `pas`, `utr_isoform`, and `regulatory_region` nodes and zero test-gene statistics in training graphs.

### 1.4 Target Leakage Protection & 1.5 Relation-Specific Masking
- **Pre-fix issue identified (Blocker resolved):** While direct leakage paths (`gene.x[:, 0]` $\leftrightarrow$ `expresses_gene`, `regulator.x[:, 0:2]` $\leftrightarrow$ `expresses_regulator`, `pas.x[:, 2:5]` $\leftrightarrow$ `utr_isoform.x[:, 2]` $\leftrightarrow$ `uses_isoform`, and `interaction_score`) were sanitized, an **indirect PAS-usage leakage path** remained: `regulatory_region.x[:, 3]` (`survival_probability`), `regulator.x[:, 5:7]` (`fractional_cis_dosage_*`), `candidate_regulation.edge_attr[:, 0]` (`sequence_or_cis_prior`), `gene.x[:, 3:6]` (SI/magnetization/entropy), and `population.x[:, 2:5]` are deterministic functions of the gene's `pas_proportion` vector $\theta_k(c)$.
- **Consolidation fix applied:** [`apply_feature_mask_with_leakage_guard`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/leakage.py#L170-L258) now sanitizes all coupled PAS-usage-derived node and edge features in `masked_data` whenever any PAS usage, isoform usage, motif survival, cis-dosage, or gene APA summary feature is masked. Furthermore, `fractional_cis_dosage_*` and `cis_prior` are now `log1p`-scaled in [`from_utrnet_directory`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py#L528-L536) so raw motif sums no longer dominate unweighted MSE.

### 1.6 Correct Handling of Reversed Edges (`rev_*`)
- Every forward relation `(src, rel, dst)` has a typed reverse relation `(dst, rev_rel, src)` created by [`_add_reverse_edges`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py#L130-L149) with identical `edge_category`, `evidence_tier_id`, and `is_candidate_only` metadata.
- Anti-leakage sanitization in [`leakage.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/leakage.py#L30-L55) and edge randomization in [`negative_controls.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/negative_controls.py#L14-L53) operate symmetrically on both `(src, rel, dst)` and `(dst, rev_rel, src)`.

### 1.7 Shared-Node Identities Across Merged Gene Graphs
- Shared node types (`population`, `regulator`, `cpa_factor`) are deduplicated by canonical string ID (`population:<celltype>`, `regulator:<name>`, `cpa:<factor>`), while gene-local node types (`gene`, `pas`, `utr_isoform`, `regulatory_region`) remain strictly disjoint across genes.

### 1.8 Effect of Evidence Tiers and Edge Categories on Learning (Ablation Study)
Evaluated on the strict 20-train $\rightarrow$ 5-test gene inductive split (`seed=42`, 25 epochs):

| Model / Ablation Variant | Held-Out Test MSE | Held-Out Test MAE | Relative MSE vs Baseline |
| --- | ---: | ---: | ---: |
| `TrivialNonGraphBaseline` (node-local MLP, no edges) | `0.0700` | `0.1797` | `0.0%` |
| `UTRNNetHeteroGNN` evaluated on **Randomized Edges** | `0.0756` | `0.2018` | `+8.0%` (worse) |
| `UTRNNetHeteroGNN` (`structural + annotation_derived` only) | `0.0574` | `0.1601` | `-18.0%` |
| `UTRNNetHeteroGNN` (`no inferred_candidate` edges) | `0.0614` | `0.1619` | `-12.3%` |
| `UTRNNetHeteroGNN` (**Evidence Tiers Ablated** `use_evidence_tiers=False`) | `0.0602` | `0.1702` | `-14.0%` |
| **`UTRNNetHeteroGNN` (Full 4 Edge Categories + Evidence Tiers)** | **`0.0521`** | **`0.1428`** | **`-25.6%` (best)** |

- **Finding:** Explicit `evidence_tier_id` embeddings and `evidence_factor` priors reduce held-out test MSE by **13.5%** (`0.0602 -> 0.0521`) and MAE by **16.1%** (`0.1702 -> 0.1428`) compared to treating all edges with uniform evidence weight.

### 1.9 Runtime and Peak GPU / CPU Memory
- **25-Gene Cohort Loading (`from_utrnet_directory` $\times 25$):** `28.79 s` wall-clock time.
- **Full Benchmark Suite Execution (6 models trained + evaluated on CUDA):** `63.88 s` total wall-clock time.
- **Peak GPU VRAM (`torch.cuda.max_memory_allocated`):** `4,741.62 MB` (`~4.63 GB`).
- **Peak CPU RSS (`ru_maxrss`):** `2,247.65 MB` (`~2.19 GB`).

### 1.10 What the 25-Gene Experiment Actually Demonstrates
- **Demonstrated:** Self-supervised masked node-feature imputation on static observational PBMC1k regulatory graphs and zero-shot inductive transfer to unseen gene subgraphs, showing that typed message passing and explicit evidence-tier weighting carry non-trivial structural and co-expression signal beyond node-local features and random graphs.
- **NOT Demonstrated:** It does **not** demonstrate perturbation response prediction, physical RBP occupancy, or temporal/causal dynamics. Perturbation response prediction is evaluated separately in the Tahoe-100M benchmark (`Task C`).

---

## 2. Merge Recommendation for PR #1

- **Recommendation:** With the three consolidation fixes applied in this branch (`log1p` cis-dosage scaling, indirect PAS-usage leakage guard in [`leakage.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/leakage.py), and shared-node gene-coupled feature masking in [`merge_hetero_graphs`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/graphs/adapter.py)), all blockers are resolved and **PR #1 is ready for merging**.
