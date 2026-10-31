# Tahoe-100M Perturbation Pilot & 4-Model Delta-Expression Benchmark (`Task C`)

## 1. Dataset Provenance & Epistemic Boundaries

- **Source Dataset:** `tahoebio/Tahoe-100M` (pinned revision `349e2a55f5538a3e73b32a06c6bdd8e9e87011e8`)
- **Versioned Selection Manifest:** [`data/tahoe/pilot_selection_manifest.json`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/data/tahoe/pilot_selection_manifest.json)
- **Pilot Cell Lines (2):**
  - `CVCL_0399` (**LoVo**, DepMap `ACH-000950`, Bowel adenocarcinoma, *KRAS* mutant)
  - `CVCL_0459` (**NCI-H460**, DepMap `ACH-000463`, Lung large-cell carcinoma, *KRAS* mutant)
- **Perturbation Conditions (4) + Matched Control:**
  - `Adagrasib` (`0.05 uM`, *KRAS* inhibitor)
  - `Afatinib` (`0.05 uM`, `0.5 uM`, `5.0 uM`, *EGFR/ERBB* family inhibitor)
  - Matched `DMSO_TF` (`0.0 uM`) vehicle controls paired by `(plate, cell_line_id)`
- **Explicit Missing Modalities (`missing_modalities: ["PAS usage", "timepoint"]`):**
  - Tahoe-100M does **not** measure 3'UTR alternative polyadenylation (`pas_proportion` / isoform usage) and does **not** provide a continuous time course.Neither modality is fabricated.

---

## 2. Sparse Extraction, Gene Mapping & Resource Footprint

Implemented in [`src/utrnnet/tahoe/adapter.py`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/adapter.py):
- **Marker Exclusion & Count Conservation:** [`expression_row_to_pairs`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/adapter.py#L84-L120) drops the leading marker pair (`genes[0]`, `expressions[0]`), maps integer `token_id`s to stable Ensembl IDs (`ENSG...`) via `gene_metadata.parquet` without symbol substitution, and validates non-negative integer counts.
- **Row-Group-Selective Pilot Extraction:** Using Parquet footer `sample` `[min, max]` statistics across shards `02325`, `02328`, `02352`, `03072`, and `03301` (`plate13`, `plate2`, `plate3`), `339` QC-passing (`pass_filter == "full"`) single cells (`14` treated sample-line groups and `8` matched `DMSO_TF` control sample-line groups) were extracted across a `40`-gene panel (`25` PBMC1k UTRNet regulatory-prior genes + `15` canonical EGFR/KRAS/MAPK pathway genes), coupled to `618` UTRNet RBP/miRNA regulators via `4,501` expression-free `candidate_regulation` prior edges and `810` gene-gene edges.

| Resource Metric | Measured Value |
| --- | ---: |
| Extracted Pilot Matrix Shape (`cells x genes`) | `339 x 40` |
| Non-Zero Entries (`nnz`) | `2,451` |
| Measured CSR Payload Bytes (`csr_payload_bytes`) | `30,772 B` (`30.1 KB`) |
| Bounded Sparse Cache Size (`npz + json`, cap `5 MB`) | `15,158 B` (`14.8 KB`) |
| Sparse Assembly Throughput | `5,823.9 cells/s` (`42,107.5 nnz/s`) |
| 4-Model Benchmark Runtime (CUDA) | `0.561 s` (Split A) / `0.229 s` (Split B) |
| Peak GPU VRAM (`observed_peak_vram_bytes`) | `89,523,712 B` (`85.4 MB`) |
| Peak Process CPU RSS (`observed_peak_ram_bytes`) | `2,081,488,896 B` (`1.94 GB`) |

---

## 3. Four-Model Fixed-Endpoint Delta-Expression Benchmark

All models are trained and evaluated on **identical sample-isolated partitions** enforced by [`validate_group_partition(group_column="sample")`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/adapter.py#L166-L175) with `200`-draw bootstrap 95% confidence intervals (`seed=42`, `top_k=10`):

1. **`zero_delta` ([`zero_delta`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/models.py#L12-L16)):** Predicts $\hat{\Delta} = 0$ relative to matched `DMSO_TF` control.
2. **`ridge` ([`RidgeDeltaPredictor`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/models.py#L20-L50)) & `regularized_mlp` ([`RegularizedMLPDeltaPredictor`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/models.py#L53-L69)):** Non-graph regularized linear and 2-layer MLP predictors over basal expression + `[drug, log1p(dose), cell_line_id]`.
3. **`gene_graph_gnn` ([`GeneGraphDeltaPredictor`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/models.py#L72-L115)):** 2-layer homogeneous message-passing GNN over gene-gene edges.
4. **`regulatory_prior_hetero_gnn` ([`RegulatoryPriorHeteroDeltaPredictor`](file:///scratch/users/megas/scPolASeq_downstream/UTRNNet/src/utrnnet/tahoe/models.py#L118-L212)):** 2-layer heterogeneous regulatory-prior GNN coupling Tahoe gene states with UTRNet static structural priors (`log1p(n_pas_sites)`, `log1p(n_utr_isoforms)`) and expression-free `("regulator", "candidate_regulation", "gene")` prior edges.

### 3.1 Split A — Sample-Isolated Replicate Split (10 Train Rows $\rightarrow$ 4 Held-Out Sample-Line Rows)

| Model | Test MSE (`95% CI`) | Test MAE (`95% CI`) | Directional Sign Agreement | Top-10 DE Gene Recovery |
| --- | --- | --- | ---: | ---: |
| `zero_delta` | `0.0861` (`[0.0307, 0.1433]`) | `0.1929` (`[0.1368, 0.2549]`) | `0.000` | `0.100` |
| `ridge` ($\alpha=1.0$) | `0.1086` (`[0.0552, 0.1672]`) | `0.2131` (`[0.1623, 0.2638]`) | `0.536` | `0.400` |
| `regularized_mlp` | `0.1132` (`[0.0550, 0.1722]`) | `0.2226` (`[0.1638, 0.2796]`) | `0.529` | `0.325` |
| `gene_graph_gnn` | `0.1045` (`[0.0514, 0.1584]`) | `0.2121` (`[0.1646, 0.2578]`) | **`0.543`** | `0.375` |
| **`regulatory_prior_hetero_gnn`** | `0.1076` (`[0.0536, 0.1619]`) | `0.2152` (`[0.1712, 0.2588]`) | `0.529` | **`0.450`** |

### 3.2 Split B — Held-Out Perturbation Condition Split (`Afatinib 0.5 uM` Held Out Completely)

| Model | Test MSE (`95% CI`) | Test MAE (`95% CI`) | Mean Pearson $r$ | Directional Sign Agreement | Top-10 DE Gene Recovery |
| --- | --- | --- | ---: | ---: | ---: |
| `zero_delta` | `0.0478` (`[0.0289, 0.0780]`) | `0.1478` (`[0.1199, 0.1860]`) | `0.000` | `0.000` | `0.150` |
| `ridge` ($\alpha=1.0$) | `0.0621` (`[0.0415, 0.0827]`) | `0.1719` (`[0.1424, 0.1989]`) | `0.108` | `0.597` | `0.250` |
| `regularized_mlp` | `0.0713` (`[0.0456, 0.0970]`) | `0.1880` (`[0.1588, 0.2171]`) | **`0.154`** | **`0.632`** | **`0.400`** |
| **`gene_graph_gnn`** | **`0.0602`** (`[0.0433, 0.0775]`) | `0.1740` (`[0.1463, 0.2017]`) | `0.008` | `0.535` | `0.325` |
| **`regulatory_prior_hetero_gnn`** | **`0.0609`** (`[0.0448, 0.0785]`) | `0.1769` (`[0.1497, 0.2024]`) | `0.064` | `0.521` | `0.200` |

---

## 4. Scientific Interpretation & Limitations

1. **Why `zero_delta` achieves lower raw MSE but zero directional power:** At small single-cell slice sizes (`339` cells across `14` treated sample-line groups), log1p-CPM pseudobulk deltas have small variance around zero (`MSE ~ 0.048–0.086`), so predicting $\hat{\Delta}=0$ minimizes squared error by shrinkage while providing `0.000` directional agreement and random (`0.10–0.15`) top-$k$ DE recovery.
2. **Value of Graph & UTRNet Regulatory Priors:** Among active predictors, `gene_graph_gnn` and `regulatory_prior_hetero_gnn` achieve the lowest MSE on both splits (`0.1045` / `0.1076` on Split A; `0.0602` / `0.0609` on Split B), and `regulatory_prior_hetero_gnn` achieves the highest top-10 DE gene recovery (`0.450`) on the replicate split (`Split A`), while `regularized_mlp` achieves the highest Pearson $r$ (`0.154`) and directional agreement (`0.632`) when interpolating an unseen dose (`Afatinib 0.5 uM` between `0.05 uM` and `5.0 uM`).
3. **Overlapping 95% Bootstrap CIs:** Confidence intervals overlap across all learned models on this bounded pilot slice (`n_test = 4` sample-line groups); expanding to all 14 plates via pre-aggregated pseudobulk shards (`metadata/pseudobulk_differential_expression`) is the recommended next step before claiming statistically significant superiority between graph architectures.
