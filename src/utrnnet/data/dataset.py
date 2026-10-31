"""Dataset and cohort loading utilities for UTRNNet."""
from __future__ import annotations

from pathlib import Path
from torch_geometric.data import HeteroData

from utrnnet.graphs.adapter import from_utrnet_directory, merge_hetero_graphs


def load_gene_cohort_graphs(
    cohort_dir: Path | str,
    gene_ids: list[str] | None = None,
    *,
    population: str | None = None,
    merge: bool = False,
    min_pas_sites: int = 2,
    strip_expression_from_candidate_edges: bool = False,
) -> list[HeteroData] | HeteroData:
    """Load a quality-controlled cohort of gene regulatory graphs from a UTRNet output root directory.

    Parameters
    ----------
    cohort_dir : Path | str
        Root output directory containing per-gene subdirectories (e.g. `outputs/pbmc1k/ENSG...`).
    gene_ids : list[str] | None
        Optional explicit list of gene directory names. If `None`, discovers all subdirectories
        containing `interaction_scores.csv` and `utr_isoforms.csv`.
    population : str | None
        Optional population filter (`celltype_corrected`).
    merge : bool
        If True, merges all loaded gene graphs into a single multi-gene `HeteroData` object.
    min_pas_sites : int
        Quality-control filter requiring at least `min_pas_sites` PAS nodes per gene.
    strip_expression_from_candidate_edges : bool
        If True, strips `interaction_score` from candidate edges at load time.
    """
    cohort_dir = Path(cohort_dir)
    if gene_ids is None:
        candidates = sorted(
            p.name
            for p in cohort_dir.iterdir()
            if p.is_dir() and (p / "interaction_scores.csv").exists() and (p / "utr_isoforms.csv").exists()
        )
    else:
        candidates = list(gene_ids)

    graphs: list[HeteroData] = []
    for gid in candidates:
        gpath = cohort_dir / gid
        if not (gpath / "interaction_scores.csv").exists():
            continue
        g = from_utrnet_directory(
            gpath,
            population=population,
            strip_expression_from_candidate_edges=strip_expression_from_candidate_edges,
        )
        if int(g["pas"].x.shape[0]) >= min_pas_sites and int(g["utr_isoform"].x.shape[0]) >= 1:
            graphs.append(g)

    if not graphs:
        raise ValueError(f"No valid QC-passing gene graphs found in {cohort_dir}")

    if merge:
        return merge_hetero_graphs(graphs)
    return graphs
