"""Loss-aware Tahoe-100M metadata, sparse-count, cache, and resource adapter (`utrnnet.tahoe`)."""
from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy import sparse

DATASET_ID = "tahoebio/Tahoe-100M"
DATASET_REVISION = "349e2a55f5538a3e73b32a06c6bdd8e9e87011e8"
CONFIGS = (
    "expression_data",
    "sample_metadata",
    "gene_metadata",
    "drug_metadata",
    "cell_line_metadata",
    "obs_metadata",
    "pseudobulk_differential_expression",
)
EXPRESSION_COLUMNS = {
    "genes",
    "expressions",
    "drug",
    "sample",
    "BARCODE_SUB_LIB_ID",
    "cell_line_id",
    "moa-fine",
    "canonical_smiles",
    "pubchem_cid",
    "plate",
}
SAMPLE_COLUMNS = {"sample", "plate", "drug", "drugname_drugconc"}
GENE_COLUMNS = {"token_id", "gene_symbol", "ensembl_id"}
CONTROL_DRUG = "DMSO_TF"


def validate_columns(frame: pd.DataFrame, required: set[str], table: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{table}: missing required columns {sorted(missing)}")


def make_gene_map(gene_metadata: pd.DataFrame) -> dict[int, str]:
    """Map Tahoe tokens to stable Ensembl gene IDs without symbol substitution."""
    validate_columns(gene_metadata, GENE_COLUMNS, "gene_metadata")
    mapping: dict[int, str] = {}
    for token, ensembl in zip(gene_metadata.token_id, gene_metadata.ensembl_id):
        token_id = int(token)
        gene_id = str(ensembl)
        if not gene_id or gene_id.lower() == "nan":
            continue
        if token_id in mapping and mapping[token_id] != gene_id:
            raise ValueError(f"token_id {token_id} maps to multiple Ensembl IDs")
        mapping[token_id] = gene_id
    if not mapping:
        raise ValueError("gene_metadata produced an empty token map")
    return mapping


def parse_concentrations(value: Any) -> list[dict[str, Any]]:
    """Parse the documented `drugname_drugconc` list while retaining units."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        parsed = value
    else:
        try:
            parsed = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            return [{"raw": str(value)}]
    if not isinstance(parsed, (list, tuple)):
        return [{"raw": str(value)}]
    output = []
    for entry in parsed:
        if isinstance(entry, (list, tuple)) and len(entry) >= 3:
            output.append({"drug": str(entry[0]), "concentration": float(entry[1]), "unit": str(entry[2])})
        else:
            output.append({"raw": str(entry)})
    return output


def expression_row_to_pairs(
    row: Mapping[str, Any],
    token_to_ensembl: Mapping[int, str],
    *,
    marker_token_id: int | None = None,
    allowed_gene_ids: set[str] | None = None,
) -> list[tuple[str, int]]:
    """Drop exactly the leading marker pair (`tokens[0]`, `values[0]`), then validate and map raw counts."""
    tokens = list(row["genes"])
    values = list(row["expressions"])
    if len(tokens) != len(values):
        raise ValueError("genes and expressions lengths differ")
    if not tokens:
        raise ValueError("expression row is empty")
    leading = int(tokens[0])
    if marker_token_id is not None and leading != marker_token_id:
        raise ValueError(f"unexpected marker token {leading}; expected {marker_token_id}")
    pairs: list[tuple[str, int]] = []
    seen: set[str] = set()
    for token, value in zip(tokens[1:], values[1:]):
        token_id = int(token)
        if token_id not in token_to_ensembl:
            if allowed_gene_ids is not None:
                continue
            raise ValueError(f"unmapped gene token {token_id}")
        gene_id = token_to_ensembl[token_id]
        if allowed_gene_ids is not None and gene_id not in allowed_gene_ids:
            continue
        count = float(value)
        if not np.isfinite(count) or count < 0 or not count.is_integer():
            raise ValueError(f"invalid raw count for token {token_id}: {value}")
        if gene_id in seen:
            raise ValueError(f"duplicate gene ID {gene_id} in one cell")
        seen.add(gene_id)
        pairs.append((gene_id, int(count)))
    return pairs


def sparse_counts(
    rows: Iterable[Mapping[str, Any]],
    gene_metadata: pd.DataFrame,
    *,
    marker_token_id: int | None = None,
    target_gene_ids: list[str] | None = None,
) -> tuple[sparse.csr_matrix, list[str], list[str]]:
    """Build a cell-by-gene CSR matrix while dropping and validating the leading marker pair."""
    token_map = make_gene_map(gene_metadata)
    if target_gene_ids is not None:
        gene_ids = list(target_gene_ids)
        allowed_set = set(gene_ids)
    else:
        gene_ids = sorted(set(token_map.values()))
        allowed_set = None
    gene_col = {gene_id: i for i, gene_id in enumerate(gene_ids)}
    cell_ids: list[str] = []
    row_ix: list[int] = []
    col_ix: list[int] = []
    values: list[int] = []
    seen_cells: set[str] = set()
    expected_marker = marker_token_id
    for row in rows:
        cell_id = str(row["BARCODE_SUB_LIB_ID"])
        if cell_id in seen_cells:
            raise ValueError(f"duplicate BARCODE_SUB_LIB_ID {cell_id}")
        seen_cells.add(cell_id)
        if expected_marker is None:
            tokens = list(row["genes"])
            if not tokens:
                raise ValueError("expression row is empty")
            expected_marker = int(tokens[0])
        cell_ix = len(cell_ids)
        cell_ids.append(cell_id)
        for gene_id, count in expression_row_to_pairs(
            row, token_map, marker_token_id=expected_marker, allowed_gene_ids=allowed_set
        ):
            row_ix.append(cell_ix)
            col_ix.append(gene_col[gene_id])
            values.append(count)
    matrix = sparse.coo_matrix(
        (np.asarray(values, dtype=np.int64), (row_ix, col_ix)),
        shape=(len(cell_ids), len(gene_ids)),
        dtype=np.int64,
    ).tocsr()
    return matrix, cell_ids, gene_ids


def validate_group_partition(
    assignments: pd.DataFrame,
    *,
    group_column: str = "sample",
    partition_column: str = "partition",
) -> None:
    """Reject any sample/replicate group that appears in more than one split (`train` vs `test`)."""
    validate_columns(assignments, {group_column, partition_column}, "partition assignments")
    leaks = assignments.groupby(group_column)[partition_column].nunique()
    if (leaks > 1).any():
        raise ValueError(f"partition leakage for groups {list(leaks[leaks > 1].index)}")


def write_sparse_cache(
    matrix: sparse.csr_matrix,
    cell_ids: list[str],
    gene_ids: list[str],
    output: Path,
    *,
    max_bytes: int,
    source_revision: str = DATASET_REVISION,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Write a sparse matrix plus identity sidecar under an explicit byte cap."""
    if max_bytes <= 0:
        raise ValueError("max_bytes must be explicitly positive")
    if matrix.shape != (len(cell_ids), len(gene_ids)):
        raise ValueError("matrix dimensions do not match the supplied identifiers")
    if len(set(cell_ids)) != len(cell_ids) or len(set(gene_ids)) != len(gene_ids):
        raise ValueError("cell and gene identifiers must be unique")
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    matrix_tmp = target.with_name(target.name + ".tmp.npz")
    sidecar = target.with_name(target.name + ".json")
    sidecar_tmp = sidecar.with_name(sidecar.name + ".tmp")
    sparse.save_npz(matrix_tmp, matrix.tocsr())
    record = {
        "contract": "cell-regulatory-state/v0.1",
        "source_dataset": DATASET_ID,
        "source_revision": source_revision,
        "matrix_path": target.name,
        "shape": list(matrix.shape),
        "cell_ids": cell_ids,
        "gene_ids": gene_ids,
        "raw_count_sum": int(matrix.sum()),
        "nonzero_count": int(matrix.nnz),
        "missing_modalities": ["PAS usage", "timepoint"],
        "metadata": dict(metadata or {}),
    }
    sidecar_tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    size = matrix_tmp.stat().st_size + sidecar_tmp.stat().st_size
    if size > max_bytes:
        matrix_tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
        raise ValueError(f"pilot cache would use {size} bytes, exceeding max_bytes={max_bytes}")
    matrix_tmp.replace(target)
    sidecar_tmp.replace(sidecar)
    return target, sidecar


def sparse_resource_report(
    matrix: sparse.spmatrix,
    *,
    elapsed_seconds: float | None = None,
    observed_peak_ram_bytes: int | None = None,
    observed_peak_vram_bytes: int | None = None,
    cache_bytes: int | None = None,
) -> dict[str, Any]:
    """Report measured CSR storage separately from optional run observations."""
    csr = matrix.tocsr()
    payload_bytes = int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)
    return {
        "shape": [int(csr.shape[0]), int(csr.shape[1])],
        "nnz": int(csr.nnz),
        "csr_payload_bytes": payload_bytes,
        "csr_payload_is_ram_peak": False,
        "observed_peak_ram_bytes": observed_peak_ram_bytes,
        "observed_peak_vram_bytes": observed_peak_vram_bytes,
        "cache_bytes": cache_bytes,
        "elapsed_seconds": elapsed_seconds,
        "cells_per_second": (
            float(csr.shape[0] / elapsed_seconds)
            if elapsed_seconds is not None and elapsed_seconds > 0
            else None
        ),
        "genes_per_second": (
            float(csr.nnz / elapsed_seconds)
            if elapsed_seconds is not None and elapsed_seconds > 0
            else None
        ),
    }


def run_smoke_fixture(fixture_path: Path | str) -> dict[str, Any]:
    """Run end-to-end structural smoke validation on `examples/tahoe_smoke_fixture.json`."""
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    gene_metadata = pd.DataFrame(fixture["gene_metadata"])
    matrix, cell_ids, gene_ids = sparse_counts(fixture["expression_records"], gene_metadata)
    expected_count = sum(
        int(value) for row in fixture["expression_records"] for value in row["expressions"][1:]
    )
    observed_count = int(matrix.sum())
    if expected_count != observed_count:
        raise ValueError("smoke fixture count conservation failed")
    return {
        "status": "pass",
        "purpose": fixture["purpose"],
        "source_dataset": fixture["dataset_id"],
        "source_revision": fixture["source_revision"],
        "cells": len(cell_ids),
        "genes": len(gene_ids),
        "nonzero_counts": int(matrix.nnz),
        "raw_count_sum_after_marker_exclusion": observed_count,
        "missing_modalities": ["PAS usage", "timepoint"],
        "biological_claim": False,
    }
