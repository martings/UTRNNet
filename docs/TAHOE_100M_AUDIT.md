# Tahoe-100M source audit (initial metadata pass)

## Fixed source revision and provenance

- Dataset: tahoebio/Tahoe-100M
- Revision inspected: 349e2a55f5538a3e73b32a06c6bdd8e9e87011e8 (HF main at inspection).
- Dataset license declared on the card: CC0-1.0.
- Provenance: Tahoe Bio describes the atlas as produced with Vevo Therapeutics' Mosaic platform; the card cites the 2025 Tahoe-100M bioRxiv preprint.
- No expression-count or pseudobulk DE shards were downloaded. The obs_metadata table was streamed with projected columns under an explicit 2,500,000,000-byte source-file ceiling; the file was not retained locally.

## Observed configuration and storage inventory

The pinned dataset card declares seven Parquet configurations, each with one train split: expression_data, sample_metadata, gene_metadata, drug_metadata, cell_line_metadata, obs_metadata, and pseudobulk_differential_expression. Expression data has 95,624,334 rows; the card reports 1,693,653,078,843 uncompressed bytes and 337,644,770,670 download bytes for that split. The Hugging Face file browser reports 429 GB total repository storage. The pseudobulk DE table is partitioned into 1,026 Parquet files and its directory is reported as 88.9 GB. These figures describe different encodings/aggregations and should not be added as if they were disjoint downloads.

## Observed expression schema

| Field | Type | Interpretation |
| --- | --- | --- |
| genes | sequence<int64> | Sparse per-cell token IDs, aligned to expressions |
| expressions | sequence<float32> | Raw count values; first element is a marker sentinel |
| drug | string | Treatment; DMSO_TF denotes vehicle |
| sample | string | Sample identifier; distinguishes replicate treatments |
| BARCODE_SUB_LIB_ID | string | Unique cell barcode/sub-library key; joins obs_metadata |
| cell_line_id | string | Cellosaurus ID |
| moa-fine | string | Contextual fine-grained MOA annotation |
| canonical_smiles | string | Compound structure |
| pubchem_cid | string | PubChem identifier; empty for DMSO |
| plate | string | Experimental plate |

Map gene tokens through gene_metadata.token_id to ensembl_id and gene_symbol. Discard the first aligned marker pair before mapping or summing. Preserve original token IDs, raw cell keys and raw counts alongside normalized fields. Ensembl IDs are stated as release 109, GRCh38.

## Matching, replication and dose

DMSO_TF controls must match both experimental plate and cell_line_id. Dose is present in sample_metadata.drugname_drugconc as a stringified list of tuples carrying compound, concentration and unit; retain the original field as well as parsed numeric value and unit. The sample identifier is the replicate unit available in the table. Cell counts measure coverage and are not biological replicates. The projected metadata scan identified a provisional candidate with matched treatment/control sample IDs and plate overlap (recorded in the selection manifest). Those candidate coverage counts combine full and minimal observations; expression-compatible full-only counts still require validation before caching.


## Observed metadata table schemas and shard layout

The dataset card documents these metadata columns:

- gene_metadata: gene_symbol, ensembl_id, token_id.
- sample_metadata: sample, plate, mean_gene_count, mean_tscp_count, mean_mread_count, mean_pcnt_mito, drug, drugname_drugconc.
- drug_metadata: drug, targets, moa-broad, moa-fine, human-approved, clinical-trials, gpt-notes-approval, canonical_smiles, pubchem_cid.
- cell_line_metadata: cell_name, Cell_ID_DepMap, Cell_ID_Cellosaur, Organ, Driver_Gene_Symbol, Driver_VarZyg, Driver_VarType, Driver_ProtEffect_or_CdnaEffect, Driver_Mech_InferDM, Driver_GeneType_DM.
- obs_metadata: plate (string), BARCODE_SUB_LIB_ID (string), sample (string), gene_count (int64), tscp_count (int64), mread_count (int64), drugname_drugconc (string), drug (string), cell_line (string), sublibrary (string), BARCODE (string), pcnt_mito (float32), S_score (float64), G2M_score (float64), phase (string), pass_filter (string), cell_name (string), and __index_level_0__ (int64).
- pseudobulk_differential_expression: gene_name (string), baseMean/log2FoldChange/lfcSE/stat/pvalue/padj (float32), plate (string), n_cells_trt/n_cells_ctrl (int64), Cell_ID_Cellosaur/Cell_ID_DepMap/drug/concentration_unit/Cell_Name_Vevo (string), concentration (float32). The card does not document the DE statistical model or contrast construction.
- Actual feature types come from the pinned dataset viewer's first-rows schemas. One schema mismatch is material: drug_metadata.pubchem_cid is float64, while expression_data.pubchem_cid is string.

At the inspected revision the Hugging Face file browser showed 3,388 expression Parquet shards under data/ (338 GB directory total), and the full repository at about 429 GB. The metadata directory was about 91.2 GB because it includes the 88.9 GB pseudobulk DE directory; that DE directory contains 1,026 Parquet shards. The directly listed metadata file sizes were approximately 2.29 GB for obs_metadata, 1.33 MB for gene_metadata, 65.6 kB for sample_metadata, 40.5 kB for drug_metadata and 19 kB for cell_line_metadata. These are browser-reported file sizes, not measured download traffic.

The source README marks the Parquet configs listed above and records the expression train split as 95,624,334 rows. The file browser's shard totals and repo storage are retained as separate storage observations, not interpreted as row counts.

## Observed metadata scan and replicate structure

A projected streaming pass over obs_metadata read 100,648,790 rows: 95,624,334 had pass_filter=full and 5,024,456 had pass_filter=minimal. The full count equals the expression_data split's 95,624,334 rows. The observation table covers 50 Cellosaurus cell-line IDs, 1,344 unique sample IDs, 380 drug labels, 1,138 drug-dose labels, 14 plates, and 67,043 sample-by-cell-line groups.

A sample ID is reused across the mixed-cell-line spheroid: a single sample/well contains multiple cell lines. Therefore the metadata index key is (sample, cell_line_id), while partition assignment must keep the source sample ID together across every cell line. Count cells within each sample-line group; do not count cells as replicate units. The expression table corresponds to pass_filter=full; minimal-filter observations are retained as a separate coverage statistic and excluded from expression-compatible n_cells.

A metadata-first candidate selection manifest is included as provisional. Its coverage values are from the complete obs_metadata table before separating full and minimal rows; it still needs expression-compatible sample-line count validation before caching or model evaluation.

## Other annotations and measurements

- drug_metadata includes targets, moa-broad, moa-fine, approval/trial annotations, SMILES and PubChem IDs. Targets and MOA labels are metadata curated/inferred with GPT-based annotation and compound-source information; they are contextual metadata, not verified direct mechanisms in the assayed cells.
- cell_line_metadata includes Cellosaurus, DepMap, lineage and driver alteration annotations.
- obs_metadata is a separate Parquet table keyed by BARCODE_SUB_LIB_ID; its full schema is recorded above. cell_line contains Cellosaurus IDs and matches expression_data.cell_line_id.
- pseudobulk_differential_expression is available as its own configuration, with 1,026 sharded Parquet files. Its feature schema is recorded above, but contrast/statistical-method provenance still requires source-method review.
- This is a fixed endpoint perturbation atlas. The source description does not provide time-course measurements or gene-editing Perturb-seq interventions.


## Access-path evaluation

Hugging Face streaming is the safe way to avoid downloading the complete expression repository, but an unfiltered scan over 3,388 shards can still consume substantial bandwidth. Projection to sample, cell-line, plate, drug, dose, QC and pass_filter columns avoids reading the large genes/expressions arrays. A streaming scan of the projected obs_metadata table was completed under a prechecked 2.5 GB source-file budget. The complete expression data (3,388 shards, 338 GB) remains untouched. Parquet row-group access could reduce transfer further if range reads and row-group statistics are useful; no expression row groups have been fetched or benchmarked.

## Implementation status at this revision

- Source audit: pinned schemas for all seven configs and file inventory recorded. The source DE contrast-generation method remains undocumented.
- Metadata scan: completed on projected obs_metadata columns; the full/minimal pass_filter split and shared sample IDs across cell lines are recorded.
- Structural smoke: implemented against a checked-in synthetic fixture; it makes no biological claim.
- Pilot selector: tested against synthetic metadata. A provisional 2-cell-line/4-dose-condition candidate is recorded; exact full-filter counts for those selected sample-line groups and expression extraction remain pending.
- Sparse cache and resource reporting: implemented with explicit cache limits; actual storage, throughput, RAM and VRAM measurements require the real pilot.
- Baselines and GNN: a zero-delta baseline, ridge evaluator and optional two-layer GNN are implemented as benchmark components. No model has been trained on Tahoe, and no comparative performance is reported.
- Shared architecture contract: field mapping is documented, but the canonical cell-regulatory-state/v0.1 schema was not in the repository tree, so strict schema validation remains pending.

## Download budget and next validation

The full expression split is far outside the pilot budget. Do not materialize it or the full pseudobulk DE configuration. The index command checks the obs_metadata file size against an explicit byte cap before streaming projected fields. The cache writer also requires an explicit pilot-cache byte cap and writes source revision, identities, count totals and missing modalities. Before the provisional conditions can be cached, validate their pass_filter=full counts and determine a bounded row-group strategy for retrieving only those cells; record actual bytes and checksums. No full-scale training is authorized before that validation.

## Architecture contract mapping

The adapter's output manifest names cell-regulatory-state/v0.1 and preserves source identities, context, perturbation, dose, raw counts, sparse features, and missing modalities. The existing UTRNet ingestion contract is utrnet-input/v0.1 and is not silently substituted. A canonical serialized schema for cell-regulatory-state/v0.1 was not present in the repository at the inspected main revision; field mapping should be reconciled when that shared schema is supplied.
