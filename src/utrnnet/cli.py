"""Command-line interface for UTRNNet (`utrnnet`)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import yaml

from utrnnet.contracts.schema import validate_cell_regulatory_state
from utrnnet.data.dataset import load_gene_cohort_graphs
from utrnnet.evaluation.metrics import run_three_way_benchmark
from utrnnet.graphs.adapter import from_graphml, from_utrnet_directory
from utrnnet.graphs.serialization import save_hetero_data
from utrnnet.training.trainer import TrainingConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="utrnnet",
        description="UTRNNet: Typed Heterogeneous Regulatory GNN extension of scPolASeq-UTRNet",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_convert = sub.add_parser("convert", help="Convert a UTRNet gene output directory or GraphML into PyG HeteroData")
    p_convert.add_argument("source", type=Path, help="UTRNet gene output directory or .graphml file")
    p_convert.add_argument("--output", "-o", type=Path, required=True, help="Output .pt file path")
    p_convert.add_argument("--population", default=None, help="Optional single population name")

    p_train = sub.add_parser("benchmark", help="Run baseline vs HeteroGNN vs negative-control benchmark from YAML config")
    p_train.add_argument("--config", "-c", type=Path, required=True, help="Path to GNN experiment YAML config")
    p_train.add_argument("--output-dir", "-o", type=Path, required=True, help="Output directory for checkpoints/metrics")

    args = parser.parse_args(argv)

    if args.command == "convert":
        if args.source.is_file() and args.source.suffix == ".graphml":
            data = from_graphml(args.source)
        else:
            data = from_utrnet_directory(args.source, population=args.population)
        report = validate_cell_regulatory_state(data, raise_on_error=True)
        save_hetero_data(data, args.output)
        print(json.dumps(asdict(report), indent=2))
        return 0

    if args.command == "benchmark":
        raw_cfg = yaml.safe_load(args.config.read_text())
        data_cfg = raw_cfg.get("data", {})
        train_cfg_dict = raw_cfg.get("training", {})

        cohort_dir = Path(data_cfg["cohort_dir"])
        gene_ids = data_cfg.get("gene_ids", None)
        merge_graphs = bool(data_cfg.get("merge_graphs", True))
        population = data_cfg.get("population", None)

        graphs = load_gene_cohort_graphs(
            cohort_dir,
            gene_ids=gene_ids,
            population=population,
            merge=merge_graphs,
        )
        tcfg = TrainingConfig(
            seed=int(train_cfg_dict.get("seed", 42)),
            hidden_channels=int(train_cfg_dict.get("hidden_channels", 64)),
            num_layers=int(train_cfg_dict.get("num_layers", 2)),
            heads=int(train_cfg_dict.get("heads", 4)),
            dropout=float(train_cfg_dict.get("dropout", 0.1)),
            lr=float(train_cfg_dict.get("lr", 3e-3)),
            weight_decay=float(train_cfg_dict.get("weight_decay", 1e-4)),
            epochs=int(train_cfg_dict.get("epochs", 30)),
            mask_ratio=float(train_cfg_dict.get("mask_ratio", 0.25)),
            device=str(train_cfg_dict.get("device", "auto")),
        )
        summary = run_three_way_benchmark(graphs, tcfg, output_dir=args.output_dir)
        print(json.dumps(asdict(summary), indent=2))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
