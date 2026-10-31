# UTRNNet (UTR Regulatory Neural Network) & scPolASeq-UTRNet

`UTRNNet` extends [`scpolaseq_utrnet`](https://github.com/martings/scpolaseq_utrnet) (initialized from upstream `main` commit `02997cb5a5e170200d1efeb42880050e3de8cfdd` with full Git history preserved and `upstream` configured as read-only) with a typed PyTorch Geometric heterogeneous regulatory graph adapter (`cell-regulatory-state/v0.1`), strict anti-leakage sanitization, and a two-layer heterogeneous message-passing encoder (`UTRNNetHeteroGNN`).

`scpolaseq_utrnet` builds exploratory, population-specific 3′UTR regulatory graphs from frozen scPolASeq PAS counts. The PBMC 1k MVP uses the completed `pbmc_1k_v3` replica and its technically concordant downstream notebook output. It does not rerun alignment, Sierra, SCAPTURE, or PAS discovery.

## UTRNNet (`utrnnet`) PyTorch Geometric Quick Start

```bash
python -m pip install -e '.[test,gnn]'
utrnnet convert /path/to/outputs/pbmc1k/ENSG00000130066 -o /tmp/sat1_state.pt
utrnnet benchmark --config configs/gnn/pbmc1k_m2_benchmark.yaml --output-dir /tmp/utrnnet_bench
python -m pytest -q
```

See:
- [Architecture & `cell-regulatory-state/v0.1` Data Contract](docs/architecture/CELL_REGULATORY_STATE_V0_1.md)
- [M1–M2 Experiment Log, Metrics & Prioritized Backlog](docs/experiments/M1_M2_EXPERIMENT_REPORT_AND_BACKLOG.md)

## PBMC 1k quick start on Deepthought

```bash
python -m pip install -e '.[test]'
export UTRNET_UPSTREAM=/path/to/completed/scPolASeq/pbmc1k
export UTRNET_DOWNSTREAM=/path/to/executed/downstream/outputs/pbmc1k
utrnet audit --config configs/pbmc1k.yaml
utrnet candidates --config configs/pbmc1k.yaml
utrnet ingest --config configs/pbmc1k.yaml --gene SAT1
utrnet build-graph --config configs/pbmc1k.yaml --gene SAT1
utrnet compare --config configs/pbmc1k.yaml --gene SAT1
utrnet render --config configs/pbmc1k.yaml --gene SAT1
utrnet build-graph --config configs/pbmc1k.yaml --gene CORO1B
utrnet render --config configs/pbmc1k.yaml --gene CORO1B
utrnet sensitivity --config configs/pbmc1k.yaml --dry-run
utrnet sensitivity --config configs/pbmc1k.yaml --gene SAT1
```

Set the two input paths to the completed scPolASeq run and its executed downstream notebook output before running the commands. The configuration rejects unset path variables. Outputs are written under `outputs/pbmc1k/<gene ID>/` and excluded from Git. The HTML explorer, static PNG, normalized CSV tables, Cytoscape JSON, GraphML, and provenance are generated locally. See [PBMC validation](docs/PBMC1K_VALIDATION.md).

SAT1 and CORO1B form the first two-gene panel because both have observed alternative PAS and compatible 3′UTR sequence models. CORO1B also has ambiguous alternative sequence models, which stay flagged. `site_retention.csv` reports sites shared across compatible PAS, restricted to some PAS, or dependent on the selected transcript model. The bundled RBP motif subset is reproducibly extracted from ATtRACT with a pinned archive checksum. The bundled MirGeneDB human mature sequences supply exact miRNA seed-match predictions. IFITM3 remains an APA observation but its highlighted proximal PAS lacks a compatible annotated 3′UTR model in this audit; it is not used to claim a regulatory switch. ZSWIM6 remains excluded pending its large-span mapping review.

## Run without Deepthought

```bash
python examples/make_fixture.py /tmp/utrnet-fixture
utrnet build-graph --config /tmp/utrnet-fixture/config.yaml
utrnet render --config /tmp/utrnet-fixture/config.yaml
python -m pytest -q
```

The fixture contains two cells, two strands, exact barcodes with `-1` suffixes, a synthetic 3′UTR motif retained only in a distal isoform, and a prior-only miRNA site. It is synthetic and makes no biological claim.

## Evidence and interpretation

Observed: Sierra UMI counts, curated cell identities, and mRNA expression. Annotation-derived: candidate transcript/PAS associations and reconstructed 3′UTR sequences. External experimental evidence: ATtRACT's ELAVL1 and ZFP36 AUUUA motif records from other contexts. Predicted: matches of those motifs and MirGeneDB-derived exact miRNA seeds to SAT1 isoforms, plus interaction scores. Missing by default: RNAplfold accessibility if the executable is absent, measured RBP/AGO occupancy at SAT1 in PBMC, and endogenous miRNA abundance. The optional miRNA genomic-site import uses a versioned TSV; both miRNA evidence modes remain prior-only without small-RNA measurements.

Scores expose usage, model weight, motif presence, accessibility, evidence factor, and abundance factor separately. Missing accessibility or abundance yields an explicitly labeled partial score. `population_metrics.csv` summarizes candidate-network size and weight concentration. The executable sensitivity command writes `sensitivity.csv` with five graph-only scenarios; `--dry-run` still reports which stages a parameter change would invalidate. The explorer displays up to eight representative regulator patterns for legibility; its site-retention and interaction tables and exported graphs retain the full candidate set. Results compare lineages within one biological sampling context; cells are not independent biological replicates.

## Repository layout

- `src/scpolaseq_utrnet/`: typed ingestion, isoform, evidence, folding, graph and visualization modules.
- `configs/pbmc1k.yaml`: reproducible local input configuration.
- `resources/are_motifs.tsv`: two checksum-pinned ATtRACT records with PubMed, assay and context columns; [resource notes](docs/REGULATORY_RESOURCES.md).
- `resources/mirgenedb3_hsa_mature.fa`: checksum-pinned human mature miRNA sequences for seed-only site predictions.
- `examples/make_fixture.py`, `tests/`: synthetic demonstration and integration checks.
- `docs/`: prior-run inventory, data contract, scientific model, current PBMC validation, and reprocessing backlog.
- `.agents/skills/`: seven task-specific scientific and engineering skills.

No large datasets, reference FASTA/GTF, BAMs, private configuration or credentials belong in Git.
