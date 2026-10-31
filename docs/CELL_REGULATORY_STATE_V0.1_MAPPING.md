# Tahoe adapter mapping to cell-regulatory-state/v0.1

The Tahoe adapter marks its selection manifest with contract version cell-regulatory-state/v0.1. The current repository main branch contains utrnet-input/v0.1 for frozen PBMC APA graph inputs; that contract is not equivalent to the shared perturbation-response contract.

| Cell-regulatory field | Tahoe source | Handling |
| --- | --- | --- |
| sample_id / replicate_id | expression_data.sample | Preserve as source replicate key. Do not infer biological independence from cell count. |
| cell_id | expression_data.BARCODE_SUB_LIB_ID | Preserve verbatim; also joins obs_metadata. |
| cell_line_id | expression_data.cell_line_id | Preserve Cellosaurus identifier; cross-reference cell_line_metadata without replacing it. |
| perturbation_id | expression_data.drug | Preserve original compound label; do not promote target annotations to causal mechanisms. |
| dose | sample_metadata.drugname_drugconc | Preserve source string and parse concentration plus unit as additional fields. |
| plate_id | expression_data.plate / sample_metadata.plate | Preserve source value; use for DMSO_TF control matching together with cell_line_id. |
| expression | expression_data genes/expressions | Raw sparse counts after dropping only the documented leading marker pair and mapping token IDs via gene_metadata. |
| gene_id | gene_metadata.ensembl_id | Human Ensembl release 109, genome build GRCh38; preserve token_id and gene_symbol as source columns. |
| perturbation annotations | drug_metadata.targets, moa-broad, moa-fine | Context only; retain annotation provenance and uncertainty. |
| APA/PAS usage | absent from Tahoe tables | Missing modality, never zero-filled or fabricated. |
| timepoint | absent from source description/schema | Missing. Treat outcomes as fixed endpoint responses. |

No canonical schema file for cell-regulatory-state/v0.1 was found on the inspected repository main tree. This document records the proposed field mapping only; exact field names, enum values and validation must be reconciled against the shared canonical contract before declaring interoperability complete.
