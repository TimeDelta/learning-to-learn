from __future__ import annotations

import argparse
import configparser
import copy
import csv
import hashlib
import itertools
import json
import logging
import math
import os
import re
import statistics
import tempfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import neat
import torch
import torch.nn as nn
from neat.reporting import BaseReporter

from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import *
from population import *
from population import _INVALID_REASON_COUNTER
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import *
from torchscript_utils import serialize_script_module

GUIDED_POPULATION_SECTION = "GuidedPopulation"


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Expected boolean value for GuidedPopulation override, got '{value}'")


GUIDED_POPULATION_FIELDS = {
    "kl_partial_slice_ratio": float,
    "kl_partial_slice_dims": int,
    "kl_partial_slice_start": int,
    "wl_kernel_loss_weight": float,
    "wl_kernel_iterations": int,
    "trainer_freeze_cycle": str,
    "trainer_freeze_verbose": _parse_bool,
}


LOG_LEVEL_CHOICES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")


def _configure_root_logger(log_level: str) -> None:
    """Initialize the root logger according to the CLI-provided level."""

    level = logging.getLevelName(log_level.upper())
    if not isinstance(level, int):
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def load_guided_population_overrides(config_path: str | os.PathLike[str]) -> dict[str, object]:
    """Parse optional GuidedPopulation overrides from the NEAT config file."""

    parser = configparser.ConfigParser()
    try:
        read_files = parser.read(config_path)
    except configparser.Error as exc:  # pragma: no cover - configparse guard
        raise ValueError(f"Failed to parse config file '{config_path}': {exc}") from exc
    if not read_files or not parser.has_section(GUIDED_POPULATION_SECTION):
        return {}
    section = parser[GUIDED_POPULATION_SECTION]
    overrides: dict[str, object] = {}
    for key, caster in GUIDED_POPULATION_FIELDS.items():
        if key not in section:
            continue
        raw_value = section.get(key, "").strip()
        if raw_value == "":
            continue
        try:
            overrides[key] = caster(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value '{raw_value}' for {key} in [{GUIDED_POPULATION_SECTION}] of {config_path}"
            ) from exc
    return overrides


def _encode_string_sequence(values):
    tokens = [str(v) for v in values if v is not None]
    if not tokens:
        return None
    hashed = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        hashed.append(int.from_bytes(digest[:4], byteorder="little") / 0xFFFFFFFF)
    return torch.tensor(hashed, dtype=torch.float32)


def _annotate_node_for_speciation(gene: NodeGene, node: torch._C.Node) -> None:
    """Attach deterministic metadata so speciation can distinguish identical graphs."""
    if node is None:
        return
    attrs = gene.dynamic_attributes
    attrs["__node_kind__"] = node.kind()
    scope = node.scopeName()
    if scope:
        attrs["__scope__"] = scope

    outputs = list(node.outputs())
    attrs["__num_outputs__"] = len(outputs)
    output_tensor = _encode_string_sequence(str(out.type()) for out in outputs)
    if output_tensor is not None:
        attrs["__output_types__"] = output_tensor

    inputs = list(node.inputs())
    attrs["__num_inputs__"] = len(inputs)
    if inputs:
        kind_tensor = _encode_string_sequence(inp.node().kind() for inp in inputs)
        if kind_tensor is not None:
            attrs["__input_kinds__"] = kind_tensor
        type_tensor = _encode_string_sequence(str(inp.type()) for inp in inputs)
        if type_tensor is not None:
            attrs["__input_types__"] = type_tensor
        getattr_types = []
        for inp in inputs:
            src_node = inp.node()
            if src_node.kind() == "prim::GetAttr" and src_node.outputsSize() > 0:
                try:
                    getattr_types.append(str(src_node.output().type()))
                except RuntimeError:
                    continue
        getattr_tensor = _encode_string_sequence(getattr_types)
        if getattr_tensor is not None:
            attrs["__getattr_output_types__"] = getattr_tensor


def create_initial_genome(config, optimizer):
    """
    Creates an initial genome that mirrors the structure of the provided TorchScript optimizer computation graph.
    """
    genome = config.genome_type(0)
    genome.serialized_module = serialize_script_module(optimizer)

    node_mapping = {}  # from TorchScript nodes to genome node keys
    next_node_id = 0
    graph_inputs = {val.node() for val in optimizer.graph.inputs()}
    for node in optimizer.graph.nodes():
        new_node_gene = NodeGene(next_node_id, node)
        new_node_gene.init_attributes(config.genome_config)
        _annotate_node_for_speciation(new_node_gene, node)
        genome.nodes[next_node_id] = new_node_gene
        node_mapping[node] = next_node_id
        next_node_id += 1

    connections = {}
    innovation = 0
    for node in optimizer.graph.nodes():
        current_key = node_mapping[node]
        for inp in node.inputs():
            producer = inp.node()
            # only create a connection if the producer is part of our mapping
            if producer in node_mapping:
                in_key = node_mapping[producer]
                key = (in_key, current_key)
                conn = ConnectionGene(key)
                conn.enabled = True
                conn.innovation = innovation
                innovation += 1
                connections[key] = conn
            elif producer not in graph_inputs or producer.kind() != "prim::Param":
                print(f"WARNING: missing mapping for input node [{producer}]")

    genome.connections = connections
    genome.optimizer = optimizer
    return genome


def override_initial_population(population, config):
    """
    Overrides the initial genomes in the population with copies of the exact initial genome.
    """
    new_population = {}
    unique_paths = []
    seen_hashes = set()
    for fname in os.listdir("computation_graphs/optimizers/"):
        if not fname.endswith(".pt"):
            continue
        path = f"computation_graphs/optimizers/{fname}"
        with open(path, "rb") as fh:
            md5_hash = hashlib.md5(fh.read()).hexdigest()
        if md5_hash in seen_hashes:
            continue
        seen_hashes.add(md5_hash)
        unique_paths.append(path)
    for key, path in zip(population.population.keys(), unique_paths):
        optimizer = torch.jit.load(path)
        new_genome = create_initial_genome(config, optimizer)
        new_genome.key = key
        new_genome.optimizer_path = path
        new_population[key] = new_genome
    population.population = new_population
    population.shared_attr_vocab.add_names(ATTRIBUTE_NAMES)
    population.species.speciate(config, population.population, population.generation)


class MLflowRunManager:
    def __init__(
        self,
        *,
        tracking_uri: str | None,
        experiment_name: str,
        run_name: str | None,
        tags: dict[str, str],
        nested: bool = False,
    ) -> None:
        try:
            import mlflow
        except ImportError as exc:  # pragma: no cover - runtime dependency injection
            raise RuntimeError(
                "MLflow logging requested but the mlflow package is not installed. "
                "Install mlflow or omit --enable-mlflow."
            ) from exc
        self._mlflow = mlflow
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run_name = run_name or f"evolution_{datetime.utcnow().isoformat(timespec='seconds')}"
        self._tags = tags or {}
        self._nested = nested
        self._active = False
        self._log_lines: list[str] = []
        self._log_dirty = False
        self._log_artifact = "logs/progress.log"

    def __enter__(self):
        if self._tracking_uri:
            self._mlflow.set_tracking_uri(self._tracking_uri)
        if self._experiment_name:
            self._mlflow.set_experiment(self._experiment_name)
        self._mlflow.start_run(run_name=self._run_name, nested=self._nested)
        self._active = True
        if self._tags:
            self._mlflow.set_tags(self._tags)
        return self

    def __exit__(self, exc_type, exc, exc_tb):
        status = "FINISHED" if exc_type is None else "FAILED"
        self.finish(status=status)
        return False

    def finish(self, status: str = "FINISHED") -> None:
        if not self._active:
            return
        self.flush_log()
        self._mlflow.end_run(status)
        self._active = False

    def log_params(self, params: dict[str, object]) -> None:
        if not self._active or not params:
            return
        serializable = {k: self._serialize_param(v) for k, v in params.items() if v is not None}
        if serializable:
            self._mlflow.log_params(serializable)

    def log_metrics(self, metrics: dict[str, object], step: int | None = None) -> None:
        if not self._active or not metrics:
            return
        numeric = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, bool):
                numeric[key] = int(value)
                continue
            try:
                num = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(num):
                numeric[key] = num
        if numeric:
            self._mlflow.log_metrics(numeric, step=step)

    def log_text(self, text: str, artifact_file: str) -> None:
        if self._active and text:
            self._mlflow.log_text(text, artifact_file)

    def log_dict(self, data: dict, artifact_file: str) -> None:
        if self._active and data:
            self._mlflow.log_dict(data, artifact_file)

    def log_artifact(self, path: str) -> None:
        if self._active and os.path.exists(path):
            self._mlflow.log_artifact(path)

    def append_log_line(self, line: str) -> None:
        if not line:
            return
        self._log_lines.append(line)
        self._log_dirty = True

    def flush_log(self) -> None:
        if not self._active or not self._log_dirty:
            return
        payload = "\n".join(self._log_lines)
        self._mlflow.log_text(payload, self._log_artifact)
        self._log_dirty = False

    @staticmethod
    def _serialize_param(value: object) -> str | float | int:
        if isinstance(value, (str, int, float)):
            return value
        if isinstance(value, bool):
            return int(value)
        return str(value)


_COMPATIBILITY_RE = re.compile(r"compatibility threshold to ([0-9]+\.?[0-9]*)", re.IGNORECASE)


class MLflowLoggingReporter(BaseReporter):
    def __init__(self, run_manager: MLflowRunManager) -> None:
        self._run = run_manager
        self._generation = 0
        self._extinction_events = 0

    def start_generation(self, generation: int):
        self._generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        fitness_values = [g.fitness for g in population.values() if getattr(g, "fitness", None) is not None]
        if not fitness_values:
            return
        best_fitness = max(fitness_values)
        worst_fitness = min(fitness_values)
        num_nodes, num_connections = _genome_complexity(best_genome)
        mean_distance, stdev_distance = _genetic_distance_stats(config, population)
        metrics = {
            "population_best_fitness": best_fitness,
            "population_worst_fitness": worst_fitness,
            "population_mean_fitness": statistics.fmean(fitness_values),
            "population_stdev_fitness": statistics.pstdev(fitness_values) if len(fitness_values) > 1 else 0.0,
            "species_count": len(getattr(species, "species", {})),
            "best_genome_num_nodes": num_nodes or 0,
            "best_genome_num_enabled_connections": num_connections or 0,
            "population_size": len(population),
        }
        threshold = getattr(getattr(config, "species_set_config", None), "compatibility_threshold", None)
        if threshold is not None:
            metrics["compatibility_threshold"] = threshold
        if mean_distance is not None:
            metrics["mean_genetic_distance"] = mean_distance
            metrics["std_genetic_distance"] = stdev_distance or 0.0
        self._run.log_metrics(metrics, step=self._generation)
        if mean_distance is not None:
            self._run.append_log_line(
                f"Mean genetic distance {mean_distance:.3f}, standard deviation {stdev_distance or 0.0:.3f}"
            )
        if threshold is not None:
            self._run.append_log_line(f"Compatibility threshold currently {threshold:.3f}")
        self._log_species_summary(population, species, threshold)
        self._run.flush_log()

    def complete_extinction(self):
        self._extinction_events += 1
        self._run.log_metrics({"event_complete_extinction": self._extinction_events}, step=self._generation)
        self._run.append_log_line("Complete extinction event detected")
        self._run.flush_log()

    def found_solution(self, config, generation, best):
        metrics = {"event_found_solution": 1, "solution_generation": generation}
        self._run.log_metrics(metrics, step=generation)
        self._run.append_log_line(f"Found solution in generation {generation}")
        self._run.flush_log()

    def end_generation(self, config, population, species):
        self._run.flush_log()

    def info(self, msg):
        if not msg:
            return
        self._run.append_log_line(f"[gen {self._generation}] {msg}")
        match = _COMPATIBILITY_RE.search(msg)
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                return
            self._run.log_metrics({"compatibility_threshold": value}, step=self._generation)
            self._run.flush_log()

    def _log_species_summary(self, population, species, threshold):
        species_dict = getattr(species, "species", {}) or {}
        if not isinstance(species_dict, dict):
            species_dict = dict(species_dict)
        summary = {
            "generation": self._generation,
            "population_size": len(population),
            "species_count": len(species_dict),
            "compatibility_threshold": threshold,
            "total_extinctions": self._extinction_events,
            "species": [],
        }
        header = [
            f"Population of {len(population)} members in {len(species_dict)} species:",
            "   ID   age  size   fitness   adj fit  stag",
            "  ====  ===  ====  =========  =======  ====",
        ]
        for sid in sorted(species_dict):
            sp = species_dict[sid]
            members = getattr(sp, "members", {}) or {}
            if not isinstance(members, dict):
                try:
                    members = dict(members)
                except Exception:
                    members = {m: None for m in members}
            size = len(members)
            age = getattr(sp, "age", None)
            fitness = getattr(sp, "fitness", None)
            adj_fitness = getattr(sp, "adjusted_fitness", None)
            last_improved = getattr(sp, "last_improved", None)
            stagnation = None
            if last_improved is not None:
                stagnation = max(0, self._generation - int(last_improved))
            entry = {
                "species_id": sid,
                "age": age,
                "size": size,
                "fitness": fitness,
                "adjusted_fitness": adj_fitness,
                "stagnation": stagnation,
            }
            summary["species"].append(entry)
            sid_display = sid if isinstance(sid, int) else str(sid)
            age_display = age if age is not None else "-"
            fit_display = f"{fitness:.3f}" if isinstance(fitness, (int, float)) else "-"
            adj_display = f"{adj_fitness:.3f}" if isinstance(adj_fitness, (int, float)) else "-"
            stag_display = stagnation if stagnation is not None else "-"
            header.append(
                f"{str(sid_display):>6}"
                f"{str(age_display):>5}"
                f"{size:6d}"
                f"{fit_display:>11}"
                f"{adj_display:>9}"
                f"{str(stag_display):>6}"
            )
        header.append(f"Total extinctions: {self._extinction_events}")
        self._run.append_log_line("\n".join(header))
        self._run.log_dict(summary, f"species/generation_{self._generation:05d}.json")


def _genome_complexity(genome):
    if genome is None:
        return (None, None)
    size_attr = getattr(genome, "size", None)
    if callable(size_attr):
        try:
            size = size_attr()
        except TypeError:
            size = None
        if isinstance(size, tuple):
            if len(size) >= 2:
                return size[0], size[1]
            if size:
                return size[0], None
        if isinstance(size, (int, float)):
            return int(size), None
    nodes = getattr(genome, "nodes", None)
    connections = getattr(genome, "connections", None)
    if nodes is not None and connections is not None:
        num_enabled = sum(1 for conn in connections.values() if getattr(conn, "enabled", False))
        return len(nodes), num_enabled
    return (None, None)


def _genetic_distance_stats(config, population):
    genomes = list(population.values())
    if len(genomes) < 2:
        return (None, None)
    genome_config = getattr(config, "genome_config", None)
    if genome_config is None:
        return (None, None)
    distances = []
    for i in range(len(genomes)):
        for j in range(i + 1, len(genomes)):
            g1 = genomes[i]
            g2 = genomes[j]
            dist_fn = getattr(g1, "distance", None)
            if dist_fn is None:
                continue
            try:
                distance = dist_fn(g2, genome_config)
            except Exception:
                continue
            if distance is None or not math.isfinite(distance):
                continue
            distances.append(float(distance))
    if not distances:
        return (None, None)
    mean = statistics.fmean(distances)
    stdev = statistics.pstdev(distances) if len(distances) > 1 else 0.0
    return (mean, stdev)


def _log_trainer_history_artifacts(run_manager, history):
    if run_manager is None or not history:
        return
    history = sorted(history, key=lambda item: item.get("step", 0))
    loss_names = sorted({name for entry in history for name in entry.get("loss_terms", {}).keys()})
    metric_loss_names = sorted({name for entry in history for name in entry.get("per_metric_losses", {}).keys()})
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        csv_path = tmp_path / "trainer_losses.csv"
        header = ["step", "generation", "epoch", "total_loss"]
        header.extend([f"loss_{name}" for name in loss_names])
        header.extend([f"metric_loss_{name}" for name in metric_loss_names])
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for entry in history:
                row = [
                    entry.get("step"),
                    entry.get("generation"),
                    entry.get("epoch"),
                    entry.get("total_loss"),
                ]
                for name in loss_names:
                    row.append(entry.get("loss_terms", {}).get(name))
                for name in metric_loss_names:
                    row.append(entry.get("per_metric_losses", {}).get(name))
                writer.writerow(row)
        run_manager.log_artifact(str(csv_path))


if __name__ == "__main__":
    from genome import OptimizerGenome

    def _parse_mlflow_tags(tag_pairs):
        tags = {}
        for raw in tag_pairs or []:
            if not raw:
                continue
            if "=" not in raw:
                raise ValueError(f"Invalid MLflow tag '{raw}'. Expected KEY=VALUE format.")
            key, value = raw.split("=", 1)
            key = key.strip()
            if not key:
                raise ValueError("Tag keys cannot be empty.")
            tags[key] = value.strip()
        return tags

    def _build_arg_parser():
        parser = argparse.ArgumentParser(description="Evolve optimizer computation graphs")
        parser.add_argument(
            "--config-file",
            default="neat-config",
            help="Path to the NEAT configuration file (default: neat-config)",
        )
        parser.add_argument(
            "--num-generations",
            type=int,
            default=1000,
            help="Number of generations to evolve (default: 1000)",
        )
        parser.add_argument(
            "--test",
            dest="test_mode",
            action="store_true",
            help=("Reduce trainer epochs per generation for quick smoke tests or continuous integration runs."),
        )
        parser.add_argument(
            "--enable-mlflow",
            action="store_true",
            help="Enable MLflow tracking for this run.",
        )
        parser.add_argument(
            "--mlflow-tracking-uri",
            type=str,
            default="http://127.0.0.1:5000",
            help="Optional MLflow tracking URI (faslls back to the default client configuration).",
        )
        parser.add_argument(
            "--mlflow-experiment",
            type=str,
            default="testing",
            help="MLflow experiment name to use when logging is enabled.",
        )
        parser.add_argument(
            "--mlflow-run-name",
            type=str,
            default=None,
            help="Optional MLflow run name; defaults to a timestamped identifier.",
        )
        parser.add_argument(
            "--mlflow-tag",
            dest="mlflow_tags",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Additional MLflow tag to attach to the run (repeatable).",
        )
        parser.add_argument(
            "--mlflow-nested",
            action="store_true",
            help="Log the run as a nested MLflow run.",
        )
        parser.add_argument(
            "--log-level",
            type=str.upper,
            choices=LOG_LEVEL_CHOICES,
            default="INFO",
            help="Python logging level for this run (default: INFO).",
        )
        parser.add_argument(
            "--max-evaluation-steps",
            type=int,
            default=None,
            metavar="N",
            help="Cap the number of task training epochs (optimizer evaluation steps) per generation.",
        )
        parser.add_argument(
            "--final-population-dir",
            type=str,
            default="artifacts/final_population",
            help=(
                "Directory where a TorchScript-friendly snapshot of the final population will be saved "
                "(default: artifacts/final_population)."
            ),
        )
        return parser

    def _parse_args():
        parser = _build_arg_parser()
        args = parser.parse_args()
        try:
            args.mlflow_tags = _parse_mlflow_tags(args.mlflow_tags)
        except ValueError as exc:  # pragma: no cover - argparse passthrough
            parser.error(str(exc))
        return args

    args = _parse_args()
    _configure_root_logger(args.log_level)
    guided_population_overrides = load_guided_population_overrides(args.config_file)

    config = neat.Config(
        OptimizerGenome, GuidedReproduction, neat.DefaultSpeciesSet, RelativeRankStagnation, args.config_file
    )
    override_keys = tuple(GUIDED_POPULATION_FIELDS.keys())
    for key in override_keys:
        value = guided_population_overrides.get(key)
        if value is None and hasattr(config, "reproduction_config"):
            value = getattr(config.reproduction_config, key, None)
        if value is not None:
            setattr(config, key, value)
    if getattr(args, "test_mode", False):
        setattr(config, "test_mode", True)
    if args.max_evaluation_steps is not None:
        max_epochs = max(1, int(args.max_evaluation_steps))
        setattr(config, "max_evaluation_steps", max_epochs)
    population = GuidedPopulation(config)

    override_initial_population(population, config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    run_context = (
        MLflowRunManager(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            tags=args.mlflow_tags,
            nested=args.mlflow_nested,
        )
        if args.enable_mlflow
        else nullcontext()
    )

    with run_context as mlflow_run:
        if mlflow_run:
            mlflow_run.log_params(
                {
                    "config_file": args.config_file,
                    "population_size": config.pop_size,
                    "num_generations": args.num_generations,
                    "fitness_threshold": getattr(config, "fitness_threshold", None),
                }
            )
            kl_slice_params = {}
            trainer = getattr(population, "trainer", None)
            if trainer is not None:
                ratio = getattr(trainer, "kl_partial_slice_ratio", None)
                dims = getattr(trainer, "kl_partial_slice_dims", None)
                start = getattr(trainer, "kl_partial_slice_start", None)
                if ratio is not None:
                    kl_slice_params["kl_partial_slice_ratio"] = ratio
                if dims is not None:
                    kl_slice_params["kl_partial_slice_dims"] = dims
                if (ratio is not None or dims is not None) and start is not None:
                    kl_slice_params["kl_partial_slice_start"] = start
            if kl_slice_params:
                mlflow_run.log_params(kl_slice_params)
            mlflow_run.log_artifact(args.config_file)
            mlflow_reporter = MLflowLoggingReporter(mlflow_run)
            population.add_reporter(mlflow_reporter)
            trainer_step_counter = itertools.count(start=1)
            trainer_last_step = {"value": 0}
            trainer_metrics_history: list[dict[str, object]] = []

            def _log_trainer_progress(**kwargs):
                generation = kwargs.get("generation", 0) or 0
                epoch = kwargs.get("epoch", 0) or 0
                total_epochs = kwargs.get("total_epochs")
                total_loss = kwargs.get("total_loss")
                loss_terms = kwargs.get("loss_terms", {})
                kl_beta = kwargs.get("kl_beta")
                per_metric_losses = kwargs.get("per_metric_losses") or {}
                active_modules = tuple(kwargs.get("active_modules") or ())
                available_modules = tuple(
                    kwargs.get("available_modules")
                    or getattr(trainer, "module_names_for_logging", ("encoder", "decoder", "predictor"))
                )
                active_label = kwargs.get("active_modules_label")
                step = next(trainer_step_counter)
                trainer_last_step["value"] = step
                metrics = {
                    "trainer_total_loss": total_loss,
                    "trainer_epoch": epoch,
                    "trainer_generation": generation,
                }
                if kl_beta is not None:
                    metrics["trainer_kl_beta"] = kl_beta
                if available_modules:
                    for module_name in available_modules:
                        metrics[f"trainer_module_active_{module_name}"] = 1 if module_name in active_modules else 0
                metrics.update({f"trainer_{name}": value for name, value in loss_terms.items()})
                filtered_metrics = {k: v for k, v in metrics.items() if v is not None}
                if filtered_metrics:
                    mlflow_run.log_metrics(filtered_metrics, step=step)
                if epoch == 1:
                    mlflow_run.log_metrics({"generation_marker": generation}, step=step)
                if per_metric_losses:
                    metric_updates = {
                        f"fitness_predictor_loss_for_{name}": value for name, value in per_metric_losses.items()
                    }
                    mlflow_run.log_metrics(metric_updates, step=step)
                loss_str = ", ".join(f"{name}={value:.4f}" for name, value in loss_terms.items()) or "none"
                if total_epochs is None:
                    epoch_header = f"Epoch {epoch}"
                else:
                    epoch_header = f"Epoch {epoch}/{total_epochs}"
                log_line = f"{epoch_header}, Loss terms per batch: [{loss_str}] (total={total_loss:.4f})"
                if active_label:
                    log_line += f" | active_modules={active_label}"
                mlflow_run.append_log_line(log_line)
                mlflow_run.flush_log()
                trainer_metrics_history.append(
                    {
                        "step": step,
                        "generation": generation,
                        "epoch": epoch,
                        "total_loss": total_loss,
                        "loss_terms": loss_terms,
                        "per_metric_losses": per_metric_losses,
                    }
                )

            population.trainer.set_progress_callback(_log_trainer_progress)

            def _log_guided_stats(stats: dict):
                if not stats:
                    return
                step = trainer_last_step.get("value", 0)
                metrics = {
                    "guided_children_requested": stats.get("requested"),
                    "guided_children_created": stats.get("accepted"),
                    "guided_children_invalid_total": stats.get("invalid_total"),
                }
                invalid_by_reason = stats.get("invalid_by_reason", {}) or {}
                for reason, count in invalid_by_reason.items():
                    metrics[f"guided_children_invalid_{reason}"] = count
                clean_metrics = {k: v for k, v in metrics.items() if v is not None}
                if clean_metrics:
                    mlflow_run.log_metrics(clean_metrics, step=step)
                generation = stats.get("generation")
                parts = ", ".join(f"{reason}={count}" for reason, count in sorted(invalid_by_reason.items()))
                summary = (
                    f"Guided offspring gen {generation}: requested={stats.get('requested', 0)}, "
                    f"created={stats.get('accepted', 0)}, invalid_total={stats.get('invalid_total', 0)}"
                )
                if parts:
                    summary += f" :: {parts}"
                mlflow_run.append_log_line(summary)
                mlflow_run.flush_log()

            population.guided_stats_callback = _log_guided_stats

            def _log_dataset_stats(stats: dict):
                if not stats:
                    return
                step = trainer_last_step.get("value", 0)
                metrics = {
                    "trainer_valid_dataset_size": stats.get("valid"),
                    "trainer_invalid_dataset_size": stats.get("invalid"),
                    "trainer_total_dataset_size": stats.get("total"),
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                if metrics:
                    mlflow_run.log_metrics(metrics, step=step)
                summary = (
                    f"Dataset sizes gen {stats.get('generation')}: "
                    f"valid={stats.get('valid', 0)}, invalid={stats.get('invalid', 0)}, total={stats.get('total', 0)}"
                )
                mlflow_run.append_log_line(summary)
                mlflow_run.flush_log()

            population.dataset_stats_callback = _log_dataset_stats

            def _log_latent_prune_event(event: dict):
                if not event:
                    return
                step = trainer_last_step.get("value", 0)
                metrics = {
                    "latent_active_dims": event.get("active_after"),
                    "latent_pruned_dims": event.get("pruned_dims"),
                    "latent_prune_epoch": event.get("epoch"),
                    "latent_prune_generation": event.get("generation"),
                }
                metrics = {k: v for k, v in metrics.items() if v is not None}
                if metrics:
                    mlflow_run.log_metrics(metrics, step=step)
                summary = (
                    f"Latent pruning gen {event.get('generation')} epoch {event.get('epoch')}: "
                    f"active_before={event.get('active_before')}, active_after={event.get('active_after')}, "
                    f"pruned={event.get('pruned_dims')}"
                )
                mlflow_run.append_log_line(summary)
                mlflow_run.flush_log()

            population.trainer.set_prune_callback(_log_latent_prune_event)

        winner = population.run(args.num_generations)
        print("\nBest genome:\n{!s}".format(winner))

        snapshot_path = None
        try:
            population_snapshot = population.snapshot_population()
            final_dir = Path(args.final_population_dir).expanduser()
            final_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            filename = f"population_gen{population.generation:05d}_{timestamp}.pt"
            snapshot_path = final_dir / filename
            torch.save(population_snapshot, snapshot_path)
            print(f"Final population snapshot saved to {snapshot_path}")
        except Exception as exc:
            snapshot_path = None
            print(f"WARNING: Failed to save final population snapshot: {exc}")

        if mlflow_run:
            num_nodes, num_connections = _genome_complexity(winner)
            final_metrics = {
                "winner_fitness": getattr(winner, "fitness", None),
                "winner_num_nodes": num_nodes or 0,
                "winner_num_enabled_connections": num_connections or 0,
                "total_generations": args.num_generations,
            }
            mlflow_run.log_metrics(final_metrics, step=args.num_generations)
            mlflow_run.log_text(str(winner), "best_genome.txt")
            if _INVALID_REASON_COUNTER:
                mlflow_run.log_dict(_INVALID_REASON_COUNTER, "invalid_offspring_summary.json")
            _log_trainer_history_artifacts(mlflow_run, trainer_metrics_history)
            if snapshot_path and snapshot_path.exists():
                mlflow_run.log_artifact(str(snapshot_path))
