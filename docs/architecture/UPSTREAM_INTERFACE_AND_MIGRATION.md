# Stable Upstream Interface (`utrnet-graph/v0.1`) & Dependency Migration Plan (`Task B`)

## 1. Architectural Ownership Boundary

| Layer | Repository Owner | Schema Contract | Responsibilities |
| --- | --- | --- | --- |
| **Biological Graph Provider** | [`martings/scpolaseq_utrnet`](https://github.com/martings/scpolaseq_utrnet) (Codex) | `utrnet-graph/v0.1` | scPolASeq/Sierra PAS ingestion, 3'UTR isoform reconstruction, ATtRACT/MirGeneDB motif scanning, survival probability $S(m;c)$, ceRNA load, CPA correlation, and GraphML/CSV export. |
| **Experimental Regulatory GNN Framework** | [`martings/UTRNNet`](https://github.com/martings/UTRNNet) (Gemini) | `cell-regulatory-state/v0.1` | Graph-to-tensor PyG adapters, anti-leakage masking, heterogeneous GNN encoders/decoders, Tahoe-100M perturbation modeling, Dengue longitudinal contracts, and Gemma explanation grounding benchmarks. |

---

## 2. Schema Mapping (`utrnet-graph/v0.1` $\leftrightarrow$ `cell-regulatory-state/v0.1`)

Implemented and strictly validated in [`src/utrnnet/contracts/utrnet_graph_v0_1.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/utrnet_graph_v0_1.py):

### 2.1 Upstream Directory Contract (`utrnet-graph/v0.1`)
- **Required CSV tables:**
  - `apa_profiles.csv`: `gene_id`, `celltype_corrected`, `site_id`, `umi_sum`, `total_gene_umi`, `pas_proportion`, `n_cells` (+ optional `si`, `magnetization_genomic`, `entropy`, `usage_ci_low`, `usage_ci_high`).
  - `utr_isoforms.csv`: `isoform_id`, `gene_id`, `site_id`, `sequence_length`, `ambiguity_status`, `model_weight` (+ optional `span_status`).
  - `regulatory_sites.csv`: `regulatory_site_id`, `isoform_id`, `site_id`, `regulator`, `regulator_class`, `genomic_start`, `genomic_end`, `accessibility`, `evidence_tier`.
  - `regulator_abundance.csv`: `population`, `regulator`, `mean_umi_per_cell`.
  - `interaction_scores.csv`: `population`, `isoform_id`, `site_id`, `regulatory_site_id`, `regulator`, `regulator_class`, `model_weight`, `accessibility`, `evidence_factor`, `evidence_tier`, `abundance_factor`, `interaction_score`, `score_status`.
- **Optional CSV tables:** `pas_sites.csv`, `cpa_regulatory_edges.csv`, `cpa_expression.csv`.

### 2.2 Bidirectional Compatibility Guarantees
Validated via [`verify_upstream_to_cell_state_compatibility`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/contracts/utrnet_graph_v0_1.py#L244-L312) and tested in [`tests/gnn/test_upstream_contract_compatibility.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/tests/gnn/test_upstream_contract_compatibility.py):
1. **Entity conservation:** Exact 1-to-1 conservation of active `population`, `gene`, `pas`, `utr_isoform`, `regulatory_region`, and `regulator` counts between `utrnet-graph/v0.1` tables and `cell-regulatory-state/v0.1` `HeteroData` stores.
2. **Evidence tier completeness:** All 12 upstream evidence tiers (including CPA tiers `expression_correlation` and `curated_literature_and_correlation`) map losslessly to `EVIDENCE_TIER_TO_ID` and `EVIDENCE_TIER_FACTORS`.
3. **Ambiguity & missingness conservation:** `ambiguity_status` (`unique`, `ambiguous_multiple_models`, `no_compatible_3utr_model`) and unmeasured miRNA abundance (`x_missing_mask[:, 0:2] == True`) survive conversion and `.pt` serialization round-trips.

---

## 3. Staged Migration Plan from Bundled `scpolaseq_utrnet` to Pinned Upstream Dependency

To preserve git provenance and avoid breaking existing workflows before external reproduction is complete, migration follows three gated stages:

- **Stage 1 (Current — Verified):**
  - Keep `src/scpolaseq_utrnet/` untouched at upstream commit `02997cb5a5e170200d1efeb42880050e3de8cfdd`.
  - Enforce `utrnet-graph/v0.1` $\leftrightarrow$ `cell-regulatory-state/v0.1` compatibility in CI (`tests/gnn/test_upstream_contract_compatibility.py`) against both synthetic fixtures and external `scpolaseq_utrnet` PBMC1k outputs (`/home/megas/agents/scpolaseq_utrnet/outputs/pbmc1k`).
  - Decouple `utrnnet.graphs.adapter` from internal `scpolaseq_utrnet` private helpers (motif survival and cis-dosage are vectorized directly over `utrnet-graph/v0.1` CSV columns).
- **Stage 2 (Pinned Git Dependency):**
  - Declare `scpolaseq-utrnet @ git+https://github.com/martings/scpolaseq_utrnet.git@<release-tag-or-sha>` in `pyproject.toml` `[project.dependencies]`.
  - Verify bit-identical `HeteroData` checksums between bundled `src/scpolaseq_utrnet` and the external installed package across the 25-gene PBMC1k cohort.
- **Stage 3 (Subtree Removal):**
  - Once Stage 2 CI passes on a tagged upstream release of `scpolaseq_utrnet`, remove the bundled `src/scpolaseq_utrnet/` directory in a dedicated PR while keeping `upstream` git history intact.
