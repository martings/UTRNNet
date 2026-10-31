"""Lossless serialization and round-trip verification for `cell-regulatory-state/v0.1` `HeteroData`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import torch
from torch_geometric.data import HeteroData

from utrnnet.contracts.schema import validate_cell_regulatory_state


def hetero_data_to_dict(data: HeteroData) -> dict[str, Any]:
    """Convert a `cell-regulatory-state/v0.1` `HeteroData` object into a pure-Python/tensor dictionary."""
    validate_cell_regulatory_state(data, raise_on_error=True)
    payload: dict[str, Any] = {
        "schema_version": data.schema_version,
        "model_name": getattr(data, "model_name", "utrnnet-v0.1"),
        "is_temporal_observation": bool(getattr(data, "is_temporal_observation", False)),
        "source_directory": getattr(data, "source_directory", ""),
        "population_filter": getattr(data, "population_filter", "all"),
        "gene_id": getattr(data, "gene_id", ""),
        "nodes": {},
        "edges": {},
    }

    for ntype in data.node_types:
        store = data[ntype]
        n_entry: dict[str, Any] = {
            "x": store.x.detach().cpu(),
            "x_missing_mask": store.x_missing_mask.detach().cpu(),
            "feature_names": list(store.feature_names),
            "node_ids": list(store.node_ids),
        }
        if ntype == "utr_isoform":
            n_entry["ambiguity_code"] = store.ambiguity_code.detach().cpu()
            n_entry["ambiguity_labels"] = list(store.ambiguity_labels)
            n_entry["span_status_code"] = store.span_status_code.detach().cpu()
            n_entry["is_unresolved_mask"] = store.is_unresolved_mask.detach().cpu()
        elif ntype == "regulator" and hasattr(store, "regulator_classes"):
            n_entry["regulator_classes"] = list(store.regulator_classes)
        payload["nodes"][ntype] = n_entry

    for etype in data.edge_types:
        src_t, rel, dst_t = etype
        key = f"{src_t}__{rel}__{dst_t}"
        estore = data[etype]
        payload["edges"][key] = {
            "edge_type": list(etype),
            "edge_category": estore.edge_category,
            "feature_names": list(estore.feature_names),
            "uses_target_expression": bool(getattr(estore, "uses_target_expression", False)),
            "edge_index": estore.edge_index.detach().cpu(),
            "edge_attr": estore.edge_attr.detach().cpu(),
            "edge_attr_missing_mask": estore.edge_attr_missing_mask.detach().cpu(),
            "evidence_tier_id": estore.evidence_tier_id.detach().cpu(),
            "evidence_tier_labels": list(estore.evidence_tier_labels),
            "is_candidate_only": estore.is_candidate_only.detach().cpu(),
        }

    return payload


def hetero_data_from_dict(payload: dict[str, Any]) -> HeteroData:
    """Reconstruct a `cell-regulatory-state/v0.1` `HeteroData` object from a serialized dictionary."""
    data = HeteroData()
    data.schema_version = payload["schema_version"]
    data.model_name = payload.get("model_name", "utrnnet-v0.1")
    data.is_temporal_observation = bool(payload.get("is_temporal_observation", False))
    data.source_directory = payload.get("source_directory", "")
    data.population_filter = payload.get("population_filter", "all")
    if payload.get("gene_id"):
        data.gene_id = payload["gene_id"]

    for ntype, n_entry in payload["nodes"].items():
        store = data[ntype]
        store.x = n_entry["x"].clone().to(torch.float32)
        store.x_missing_mask = n_entry["x_missing_mask"].clone().to(torch.bool)
        store.feature_names = list(n_entry["feature_names"])
        store.node_ids = list(n_entry["node_ids"])
        store.node_id_to_index = {nid: i for i, nid in enumerate(store.node_ids)}
        if ntype == "utr_isoform":
            store.ambiguity_code = n_entry["ambiguity_code"].clone().to(torch.long)
            store.ambiguity_labels = list(n_entry["ambiguity_labels"])
            store.span_status_code = n_entry["span_status_code"].clone().to(torch.long)
            store.is_unresolved_mask = n_entry["is_unresolved_mask"].clone().to(torch.bool)
        elif ntype == "regulator" and "regulator_classes" in n_entry:
            store.regulator_classes = list(n_entry["regulator_classes"])

    for _, e_entry in payload["edges"].items():
        etype = tuple(e_entry["edge_type"])
        estore = data[etype]
        estore.edge_category = e_entry["edge_category"]
        estore.feature_names = list(e_entry["feature_names"])
        estore.uses_target_expression = bool(e_entry.get("uses_target_expression", False))
        estore.edge_index = e_entry["edge_index"].clone().to(torch.long)
        estore.edge_attr = e_entry["edge_attr"].clone().to(torch.float32)
        estore.edge_attr_missing_mask = e_entry["edge_attr_missing_mask"].clone().to(torch.bool)
        estore.evidence_tier_id = e_entry["evidence_tier_id"].clone().to(torch.long)
        estore.evidence_tier_labels = list(e_entry["evidence_tier_labels"])
        estore.is_candidate_only = e_entry["is_candidate_only"].clone().to(torch.bool)

    validate_cell_regulatory_state(data, raise_on_error=True)
    return data


def save_hetero_data(data: HeteroData, path: Path | str) -> Path:
    """Save a `HeteroData` graph to disk along with a human-readable `.meta.json` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = hetero_data_to_dict(data)
    torch.save(payload, path)

    meta = {
        "schema_version": data.schema_version,
        "model_name": getattr(data, "model_name", "utrnnet-v0.1"),
        "is_temporal_observation": bool(getattr(data, "is_temporal_observation", False)),
        "node_types": {nt: len(data[nt].node_ids) for nt in data.node_types},
        "edge_types": {
            f"{s}__{r}__{d}": int(data[(s, r, d)].edge_index.shape[1])
            for (s, r, d) in data.edge_types
        },
    }
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def load_hetero_data(path: Path | str) -> HeteroData:
    """Load and validate a serialized `HeteroData` graph from disk."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return hetero_data_from_dict(payload)
