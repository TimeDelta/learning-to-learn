#!/usr/bin/env python3
"""Render computation graph visualizations from a saved population snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from population_visualizer import (
    GraphvizNotFoundError,
    RenderContext,
    build_dot_graph,
    find_latest_snapshot,
    load_population_snapshot,
    render_with_graphviz,
    save_summary,
    select_snapshot_entries,
    write_dot_file,
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
        help="Directory where DOT/PNG artifacts will be written.",
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
        "--format",
        choices=["png", "pdf", "svg", "dot"],
        default="png",
        help="Image format to render (dot skips Graphviz rendering).",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz engine to invoke (dot, fdp, neato, ...).",
    )
    parser.add_argument(
        "--rankdir",
        choices=["LR", "TB"],
        default="LR",
        help="Graphviz rank direction (LR = left-to-right, TB = top-to-bottom).",
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
        "--skip-render",
        action="store_true",
        help="Only emit DOT files even when --format requests an image.",
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
    graphviz_error_reported = False
    for rank, entry in enumerate(entries, start=1):
        context = RenderContext(generation=generation, rank=rank, task=task)
        dot_source = build_dot_graph(
            entry,
            context=context,
            max_attr_lines=max(args.max_attr_lines, 0),
            max_attr_value_chars=max(args.max_attr_chars, 8),
            rankdir=args.rankdir,
        )
        base_name = _format_entry_basename(entry, generation, rank)
        dot_path = write_dot_file(dot_source, output_dir / f"{base_name}.dot")

        image_path = None
        if not args.skip_render and args.format != "dot":
            try:
                image_path = render_with_graphviz(dot_path, fmt=args.format, engine=args.engine)
            except GraphvizNotFoundError as exc:
                if not graphviz_error_reported:
                    print(f"warning: {exc}; only DOT files were written.", file=sys.stderr)
                    graphviz_error_reported = True
            except RuntimeError as exc:
                print(f"warning: failed to render {dot_path.name}: {exc}", file=sys.stderr)

        render_records.append(
            {
                "genome_id": entry.get("genome_id"),
                "species_id": entry.get("species_id"),
                "fitness": entry.get("fitness"),
                "dot": dot_path.name,
                "image": image_path.name if image_path else None,
                "invalid_graph": entry.get("invalid_graph"),
                "invalid_reason": entry.get("invalid_reason"),
            }
        )
        print(f"Rendered genome {entry.get('genome_id')} to {dot_path.name}")

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
