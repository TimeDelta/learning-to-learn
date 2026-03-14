import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import neat
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genome import OptimizerGenome
from graph_builder import rebuild_and_script
from population import GuidedPopulation
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import GuidedReproduction


def _build_population(config_path: str) -> GuidedPopulation:
    config = neat.Config(
        OptimizerGenome,
        GuidedReproduction,
        neat.DefaultSpeciesSet,
        RelativeRankStagnation,
        config_path,
    )
    return GuidedPopulation(config)


def _load_graph_payload(path: Path) -> tuple[dict, dict, Optional[torch.Tensor]]:
    payload = torch.load(path, map_location="cpu")
    metadata: dict = {}
    latent_vector: Optional[torch.Tensor] = None
    if isinstance(payload, dict) and "graph_dict" in payload:
        metadata = payload.get("metadata", {}) or {}
        latent_vector = payload.get("latent_vector")
        graph_dict = payload.get("graph_dict") or {}
    else:
        graph_dict = payload
    if latent_vector is not None and not isinstance(latent_vector, torch.Tensor):
        latent_vector = torch.as_tensor(latent_vector, dtype=torch.float32)
    return graph_dict, metadata, latent_vector


def _tensor_from_value(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return torch.as_tensor(value)


def _wl_features(graph_dict: dict, iterations: int = 2) -> Optional[Dict[str, float]]:
    node_types_val = graph_dict.get("node_types")
    edge_index_val = graph_dict.get("edge_index")
    if node_types_val is None or edge_index_val is None:
        return None
    node_types = _tensor_from_value(node_types_val).long().view(-1)
    if node_types.numel() == 0:
        return None
    edge_index = _tensor_from_value(edge_index_val).long()
    if edge_index.dim() == 1:
        edge_index = edge_index.view(2, -1)
    num_nodes = int(node_types.numel())
    adjacency: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        for src, dst in edge_index.t().tolist():
            if 0 <= src < num_nodes:
                adjacency[src].append(dst)
    colors = [f"t{int(label)}" for label in node_types.tolist()]
    for _ in range(max(1, iterations)):
        next_colors: List[str] = []
        for idx in range(num_nodes):
            neigh = sorted(colors[n] for n in adjacency[idx] if 0 <= n < num_nodes)
            combo = colors[idx] + "|" + "|".join(neigh)
            next_colors.append(combo)
        colors = next_colors
    counter: Dict[str, int] = {}
    for color in colors:
        counter[color] = counter.get(color, 0) + 1
    if not counter:
        return None
    total = float(sum(counter.values()))
    probs = torch.tensor([count / total for count in counter.values()], dtype=torch.float32)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum().item())
    max_bin = float(probs.max().item())
    return {
        "num_colors": float(len(counter)),
        "entropy": entropy,
        "max_bin": max_bin,
    }


def _aggregate_wl(records: Sequence[tuple[bool, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    def _mean(values: List[float]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    inactive_stats: Dict[str, List[float]] = {}
    active_stats: Dict[str, List[float]] = {}
    for inactive, stats in records:
        bucket = inactive_stats if inactive else active_stats
        for key, value in stats.items():
            bucket.setdefault(key, []).append(value)
    summary = {
        "inactive": {key: _mean(vals) for key, vals in inactive_stats.items()},
        "active": {key: _mean(vals) for key, vals in active_stats.items()},
    }
    return summary


def _stack_latents(entries: Sequence[Dict]) -> Optional[torch.Tensor]:
    latents = [entry["latent"] for entry in entries if entry.get("latent") is not None]
    if not latents:
        return None
    dims = {vec.numel() for vec in latents}
    if len(dims) != 1:
        raise ValueError("Latent vectors do not share the same dimensionality")
    return torch.stack(latents, dim=0)


def _pca(latents: torch.Tensor, components: int) -> tuple[torch.Tensor, torch.Tensor]:
    if latents.size(0) < 2:
        raise ValueError("Need at least two latents for PCA")
    components = max(1, min(components, latents.size(1)))
    centered = latents - latents.mean(dim=0, keepdim=True)
    u, s, v = torch.linalg.svd(centered, full_matrices=False)
    basis = v[:components].t()
    coords = centered @ basis
    variances = s[:components] ** 2
    total = (s**2).sum().clamp_min(1e-12)
    explained = variances / total
    return coords, explained


def _write_pca_csv(csv_path: Path, entries: Sequence[Dict], coords: torch.Tensor) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["path", "inactive", "generation"] + [f"pc{idx+1}" for idx in range(coords.size(1))]
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for entry, row in zip(entries, coords.tolist()):
            metadata = entry.get("metadata") or {}
            generation = metadata.get("generation")
            writer.writerow([entry.get("path"), int(entry.get("inactive", False)), generation, *row])


def _summarize_latents(entries: Sequence[Dict], components: int, csv_path: Optional[Path]):
    latents_tensor = _stack_latents(entries)
    if latents_tensor is None:
        return None
    inactive_latents = [
        entry["latent"] for entry in entries if entry.get("inactive") and entry.get("latent") is not None
    ]
    active_latents = [
        entry["latent"] for entry in entries if not entry.get("inactive") and entry.get("latent") is not None
    ]
    inactive = torch.stack(inactive_latents) if inactive_latents else None
    active = torch.stack(active_latents) if active_latents else None
    summary = {
        "total": latents_tensor.size(0),
        "dimension": latents_tensor.size(1),
    }
    if inactive is not None:
        inactive_center = inactive.mean(dim=0)
        summary["inactive_mean_norm"] = float(inactive_center.norm().item())
        summary["inactive_spread"] = float((inactive - inactive_center).norm(dim=1).mean().item())
    if active is not None:
        active_center = active.mean(dim=0)
        summary["active_mean_norm"] = float(active_center.norm().item())
        summary["active_spread"] = float((active - active_center).norm(dim=1).mean().item())
    if inactive is not None and active is not None:
        summary["centroid_distance"] = float((inactive.mean(dim=0) - active.mean(dim=0)).norm().item())
    if latents_tensor.size(0) >= 2:
        coords, explained = _pca(latents_tensor, components)
        summary["pca_explained_variance_ratio"] = explained.tolist()
        if csv_path is not None:
            entries_with_latent = [entry for entry in entries if entry.get("latent") is not None]
            _write_pca_csv(csv_path, entries_with_latent, coords)
    return summary


def _generation_summary(counts: Counter) -> Dict[str, int]:
    items = sorted((gen, count) for gen, count in counts.items() if gen is not None)
    return {str(gen): count for gen, count in items}


def _counterfactual_repair(population: GuidedPopulation, graph_dict: dict, runs: int) -> float:
    successes = 0
    for attempt in range(runs):
        probe = population._clone_graph_dict(graph_dict, include_history=False)
        if probe is None:
            continue
        population._prepare_decoded_graph_dict(probe)
        population._repair_graph_dict(probe)
        optimizer = rebuild_and_script(probe, population.config.genome_config, key=attempt)
        if optimizer and population._optimizer_updates_parameters(optimizer, check_steps=2):
            successes += 1
    return successes / max(1, runs)


def analyze_samples(
    population: GuidedPopulation,
    samples_dir: Path,
    limit: int,
    verbose: bool,
    delete_after: bool,
    latent_components: int,
    latent_csv: Optional[Path],
    counterfactual_runs: int,
    wl_iterations: int,
):
    paths = sorted(samples_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if limit > 0:
        paths = paths[:limit]
    processed = 0
    inactive = 0
    records: List[Dict] = []
    wl_records: List[tuple[bool, Dict[str, float]]] = []
    generation_counts: Counter = Counter()
    counterfactual_scores: List[float] = []

    for idx, path in enumerate(paths):
        graph_dict, metadata, latent_vector = _load_graph_payload(path)
        would_be_inactive = population._graph_dict_would_be_inactive(graph_dict, key=idx)
        processed += 1
        if would_be_inactive:
            inactive += 1
        entry = {
            "path": str(path),
            "inactive": bool(would_be_inactive),
            "metadata": metadata,
            "latent": latent_vector,
        }
        records.append(entry)
        generation = metadata.get("generation")
        if generation is not None:
            generation_counts[int(generation)] += 1
        wl_stats = _wl_features(graph_dict, iterations=wl_iterations)
        if wl_stats:
            wl_records.append((would_be_inactive, wl_stats))
        if verbose:
            prefix = f"gen{generation}_child{metadata.get('child_index')}" if generation is not None else path.name
            status = "inactive" if would_be_inactive else "ok"
            print(f"[{prefix}] {status}")
        if counterfactual_runs > 0 and would_be_inactive:
            score = _counterfactual_repair(population, graph_dict, counterfactual_runs)
            counterfactual_scores.append(score)
            entry["counterfactual_success_rate"] = score
        if delete_after:
            try:
                path.unlink()
            except OSError:
                pass

    latent_summary = _summarize_latents(records, latent_components, latent_csv) if records else None
    wl_summary = _aggregate_wl(wl_records) if wl_records else None
    counterfactual_summary = None
    if counterfactual_scores:
        counterfactual_summary = {
            "samples": len(counterfactual_scores),
            "mean_success": float(sum(counterfactual_scores) / len(counterfactual_scores)),
            "max_success": float(max(counterfactual_scores)),
            "min_success": float(min(counterfactual_scores)),
        }

    summary = {
        "samples_dir": str(samples_dir),
        "processed": processed,
        "inactive": inactive,
        "fraction_inactive": (inactive / processed) if processed else 0.0,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds"),
        "generation_histogram": _generation_summary(generation_counts),
        "latent_analysis": latent_summary,
        "wl_analysis": wl_summary,
        "counterfactual": counterfactual_summary,
    }
    return summary, records


def main():
    parser = argparse.ArgumentParser(description="Analyze buffered pre-repair optimizer graphs.")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("debug_guided_offspring") / "pre_repair_samples",
        help="Directory containing pre-repair graph samples (default: debug_guided_offspring/pre_repair_samples)",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="neat-config",
        help="Path to the NEAT config file used for the run (default: neat-config)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of samples to analyze (0 = all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample status instead of summary only.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete samples after they are analyzed.",
    )
    parser.add_argument(
        "--latent-components",
        type=int,
        default=2,
        help="Number of PCA components to compute for latent analysis (default: 2)",
    )
    parser.add_argument(
        "--latent-csv",
        type=Path,
        default=None,
        help="Optional path to export PCA coordinates (CSV).",
    )
    parser.add_argument(
        "--counterfactual-repair",
        type=int,
        default=0,
        metavar="N",
        help="If >0, re-run the repair hook N times for each inactive sample to estimate salvage probability.",
    )
    parser.add_argument(
        "--wl-iterations",
        type=int,
        default=2,
        help="WL subtree iterations to use when summarizing structure (default: 2)",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path to write the analysis summary as JSON.",
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir
    if not samples_dir.exists():
        raise SystemExit(f"Samples directory '{samples_dir}' does not exist")

    population = _build_population(args.config_file)
    summary, records = analyze_samples(
        population,
        samples_dir,
        args.limit,
        args.verbose,
        args.delete,
        args.latent_components,
        args.latent_csv,
        args.counterfactual_repair,
        args.wl_iterations,
    )

    print(
        f"Analyzed {summary['processed']} samples from {samples_dir}: "
        f"{summary['inactive']} would have been inactive pre-repair."
    )
    latent_analysis = summary.get("latent_analysis")
    if latent_analysis:
        centroid = latent_analysis.get("centroid_distance")
        if centroid is not None:
            print(f"Latent centroid distance (inactive vs active): {centroid:.4f}")
    wl_analysis = summary.get("wl_analysis")
    if wl_analysis:
        inactive_entropy = wl_analysis.get("inactive", {}).get("entropy")
        active_entropy = wl_analysis.get("active", {}).get("entropy")
        if inactive_entropy is not None and active_entropy is not None:
            print(f"WL entropy inactive/active: {inactive_entropy:.3f}/{active_entropy:.3f}")
    counter_summary = summary.get("counterfactual")
    if counter_summary:
        print(
            "Counterfactual repair success rate (mean/min/max): "
            f"{counter_summary['mean_success']:.3f}/"
            f"{counter_summary['min_success']:.3f}/"
            f"{counter_summary['max_success']:.3f}"
        )

    if args.summary_json:
        payload = dict(summary)
        if not args.verbose:
            payload.pop("records", None)
        args.summary_json.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
