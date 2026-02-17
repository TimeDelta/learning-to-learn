#!/usr/bin/env python3
"""Render computation graph visualizations from a saved population snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from population import DECODED_GRAPH_DICT_KEY, REPAIRED_GRAPH_DICT_KEY
from population_visualizer import (
    RenderContext,
    build_mermaid_graph,
    find_latest_snapshot,
    load_population_snapshot,
    save_summary,
    select_snapshot_entries,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Path to a population snapshot .pt file. Defaults to the newest file under --snapshot-dir.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("artifacts/final_population"),
        help="Directory to search when --snapshot is not specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_population/viz"),
        help="Directory where Mermaid artifacts will be written.",
    )
    parser.add_argument("--top-k", type=int, default=16, help="Maximum graphs to render (0 = all).")
    parser.add_argument(
        "--genome-id",
        dest="genome_ids",
        action="append",
        type=int,
        default=[],
        help="Specific genome id to render (repeatable).",
    )
    parser.add_argument(
        "--include-invalid",
        action="store_true",
        help="Include genomes marked as invalid in the snapshot.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["fitness", "genome_id"],
        default="fitness",
        help="Ordering applied before taking --top-k entries.",
    )
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB"],
        default="LR",
        help="Mermaid rank direction (LR = left-to-right, TB = top-to-bottom).",
    )
    parser.add_argument(
        "--max-attr-lines",
        type=int,
        default=4,
        help="Maximum attribute key/value lines per node label.",
    )
    parser.add_argument(
        "--max-attr-chars",
        type=int,
        default=36,
        help="Maximum characters for each attribute summary line.",
    )
    parser.add_argument(
        "--variants",
        choices=["repaired", "decoded", "both"],
        default="repaired",
        help="Which graph variants to export for each genome.",
    )
    parser.add_argument(
        "--summary-name",
        type=str,
        default="summary.json",
        help="Filename for the JSON manifest written alongside the renders.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable writing a JSON summary manifest.",
    )
    return parser.parse_args()


def _resolve_snapshot(args: argparse.Namespace) -> Path:
    if args.snapshot is not None:
        return args.snapshot
    latest = find_latest_snapshot(args.snapshot_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No snapshot files found under {args.snapshot_dir}. Provide --snapshot to select a file explicitly."
        )
    return latest


def _format_entry_basename(entry: dict, generation: Optional[int], rank: int) -> str:
    genome_id = entry.get("genome_id")
    genome_str = f"g{int(genome_id):05d}" if genome_id is not None else "gXXXX"
    if isinstance(generation, int) and generation >= 0:
        gen_str = f"gen{generation:05d}"
    else:
        gen_str = "genNA"
    return f"{gen_str}_rank{rank:02d}_{genome_str}"


def main() -> int:
    args = _parse_args()
    try:
        snapshot_path = _resolve_snapshot(args)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    snapshot = load_population_snapshot(snapshot_path)
    entries = select_snapshot_entries(
        snapshot,
        top_k=args.top_k if args.top_k > 0 else None,
        genome_ids=args.genome_ids,
        include_invalid=args.include_invalid,
        sort_by=args.sort_by,
    )
    if not entries:
        print("No entries matched the provided filters.", file=sys.stderr)
        return 1

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    generation = snapshot.get("generation")
    task = snapshot.get("task")

    render_records: List[dict] = []
    for rank, entry in enumerate(entries, start=1):
        context = RenderContext(generation=generation, rank=rank, task=task)
        base_name = _format_entry_basename(entry, generation, rank)
        base_graph = entry.get("graph") or {}
        decoded_graph = base_graph.get(DECODED_GRAPH_DICT_KEY)
        repaired_graph = base_graph.get(REPAIRED_GRAPH_DICT_KEY) or base_graph

        variants: List[Tuple[str, Mapping[str, Any]]] = []
        if args.variants in {"decoded", "both"} and decoded_graph is not None:
            variants.append(("decoded", decoded_graph))
        if args.variants in {"repaired", "both"}:
            variants.append(("repaired", repaired_graph))
        if not variants:
            variants.append(("repaired", repaired_graph))

        mermaid_files: List[str] = []
        for suffix, graph_payload in variants:
            mermaid_entry = dict(entry)
            mermaid_entry["graph"] = graph_payload
            mermaid_source = build_mermaid_graph(
                mermaid_entry,
                context=context,
                max_attr_lines=max(args.max_attr_lines, 0),
                max_attr_value_chars=max(args.max_attr_chars, 8),
                rankdir=args.rankdir,
            )
            mermaid_path = output_dir / f"{base_name}_{suffix}.mmd"
            mermaid_path.write_text(mermaid_source)
            mermaid_files.append(mermaid_path.name)

        render_records.append(
            {
                "genome_id": entry.get("genome_id"),
                "species_id": entry.get("species_id"),
                "fitness": entry.get("fitness"),
                "mermaid": mermaid_files,
                "invalid_graph": entry.get("invalid_graph"),
                "invalid_reason": entry.get("invalid_reason"),
            }
        )
        printable = ", ".join(mermaid_files)
        print(f"Rendered genome {entry.get('genome_id')} to {printable}")

    if not args.no_summary:
        summary_path = output_dir / args.summary_name
        save_summary(
            entries,
            summary_path,
            extra={
                "snapshot_path": str(snapshot_path),
                "output_dir": str(output_dir),
                "generation": generation,
                "task": task,
                "artifacts": render_records,
            },
        )
        print(f"Wrote summary manifest to {summary_path}")

    print(f"Done. Artifacts stored in {output_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
