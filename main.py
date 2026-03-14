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
from warnings import warn

import neat
import torch
import torch.nn as nn
from neat.reporting import BaseReporter

from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import *
from graph_ir import export_script_module_to_graph_ir
from loop_blocks import (
    DEFAULT_BLOCK_PAYLOAD_VALUE_DIM,
    encode_block_payload,
    register_graph_blocks,
    snapshot_registry,
)
from population import *
from population import _INVALID_REASON_COUNTER
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import *
from torchscript_utils import serialize_script_module
from utility import log_timing

GUIDED_POPULATION_SECTION = "GuidedPopulation"


logger = logging.getLogger(__name__)


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
    "repair_randomize_connections": _parse_bool,
    "repair_random_seed": int,
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

    graph_ir, module_state = export_script_module_to_graph_ir(optimizer)
    node_block_map = register_graph_blocks(graph_ir)
    block_ids = sorted({block_id for ids in node_block_map.values() for block_id in ids})
    block_registry = snapshot_registry(block_ids)
    payload_cap = getattr(config, "attr_value_max_dim", DEFAULT_BLOCK_PAYLOAD_VALUE_DIM)
    try:
        payload_cap = int(payload_cap)
    except (TypeError, ValueError):
        payload_cap = DEFAULT_BLOCK_PAYLOAD_VALUE_DIM
    payload_cap = max(DEFAULT_BLOCK_PAYLOAD_VALUE_DIM, payload_cap)

    graph_nodes = list(optimizer.graph.nodes())
    node_index_map = {node: idx for idx, node in enumerate(graph_nodes)}

    def _register_node(node_key: int, ts_node: torch._C.Node | None) -> None:
        new_node_gene = NodeGene(node_key, ts_node)
        new_node_gene.init_attributes(config.genome_config)
        if ts_node is not None:
            _annotate_node_for_speciation(new_node_gene, ts_node)
            node_idx = node_index_map.get(ts_node)
            if node_idx is not None:
                block_ids_for_node = node_block_map.get(node_idx, [])
                graph_block_attrs: Dict[str, Any] | None = None
                for block_pos, block_id in enumerate(block_ids_for_node):
                    if graph_block_attrs is None:
                        graph_block_attrs = _graph_attrs_for_node(node_key)
                    ref_name = f"{BLOCK_REF_ATTR_PREFIX}{block_pos}"
                    ref_tensor = torch.tensor([float(block_id)], dtype=torch.float32)
                    new_node_gene.dynamic_attributes[ref_name] = ref_tensor
                    graph_block_attrs[ref_name] = ref_tensor
                    payload = block_registry.get(block_id)
                    if payload:
                        payload_name = f"{BLOCK_PAYLOAD_ATTR_PREFIX}{block_pos}"
                        try:
                            encoded = encode_block_payload(payload, max_value_dim=payload_cap)
                        except ValueError as exc:
                            warn(f"Skipping block payload encoding for node {node_key}: {exc}")
                        else:
                            # stash the actual block body alongside the reference so the
                            # decoder/repair path can round-trip nested control flow
                            # without consulting external templates
                            new_node_gene.dynamic_attributes[payload_name] = encoded
                            graph_block_attrs[payload_name] = encoded
        genome.nodes[node_key] = new_node_gene

    genome = config.genome_type(0)
    genome.serialized_module = serialize_script_module(optimizer)
    module_type = optimizer._c._type().qualified_name() if hasattr(optimizer._c._type(), "qualified_name") else None
    genome.graph_dict = {
        "graph_ir": graph_ir,
        "module_state": module_state,
        "module_type": module_type,
        "block_registry": block_registry,
    }

    node_mapping = {}  # from TorchScript nodes to genome node keys
    node_attributes: List[Dict[str, Any]] = genome.graph_dict.setdefault("node_attributes", [])
    node_attr_bindings: Dict[int, Dict[str, Any]] = {}
    configured_input_keys = list(getattr(config.genome_config, "input_keys", []))
    configured_output_keys = list(getattr(config.genome_config, "output_keys", []))
    slot_node_candidates: set[int] = set()
    for key in itertools.chain(configured_input_keys, configured_output_keys):
        try:
            slot_node_candidates.add(int(key))
        except (TypeError, ValueError):
            continue

    def _ensure_attr_slot(slot_index: int) -> Dict[str, Any]:
        while len(node_attributes) <= slot_index:
            node_attributes.append({})
        attrs = node_attributes[slot_index]
        if not isinstance(attrs, dict):
            attrs = {}
            node_attributes[slot_index] = attrs
        return attrs

    def _bind_slot_to_node(slot_index: int, node_key: int) -> Dict[str, Any]:
        attrs = _ensure_attr_slot(slot_index)
        existing = node_attr_bindings.get(node_key)
        if existing is None:
            node_attr_bindings[node_key] = attrs
            return attrs
        if existing is attrs:
            return attrs
        # Preserve any previously recorded metadata on both dicts.
        for key, value in attrs.items():
            if key not in existing:
                existing[key] = value
        node_attributes[slot_index] = existing
        node_attr_bindings[node_key] = existing
        return existing

    def _graph_attrs_for_node(node_key: int) -> Dict[str, Any]:
        attrs = node_attr_bindings.get(node_key)
        if attrs is None:
            attrs = {}
            node_attr_bindings[node_key] = attrs
            if node_key not in slot_node_candidates:
                node_attributes.append(attrs)
        return attrs

    graph_input_values = list(optimizer.graph.inputs())
    graph_inputs = {val.node() for val in graph_input_values}

    user_arg_nodes: List[torch._C.Node] = []
    for value in graph_input_values:
        ts_node = value.node()
        if ts_node is None:
            continue
        try:
            type_repr = str(value.type())
        except RuntimeError:
            type_repr = ""
        if type_repr.startswith("__torch__.") and value.debugName() == "self":
            # Skip the implicit module "self" argument.
            continue
        user_arg_nodes.append(ts_node)
        if len(user_arg_nodes) >= len(configured_input_keys):
            break

    if len(user_arg_nodes) < len(configured_input_keys):
        warn(
            "TorchScript graph exposes fewer inputs than configured; missing pin roles may persist "
            f"({len(user_arg_nodes)} found vs {len(configured_input_keys)} expected)."
        )

    for slot_index, (pin_key, ts_node) in enumerate(zip(configured_input_keys, user_arg_nodes)):
        try:
            key = int(pin_key)
        except (TypeError, ValueError):
            continue
        _register_node(key, ts_node)
        node_mapping[ts_node] = key
        attrs = _bind_slot_to_node(slot_index, key)
        attrs["pin_role"] = "input"
        attrs["pin_slot_index"] = slot_index

    output_producers: List[torch._C.Node] = []
    for value in optimizer.graph.outputs():
        producer = value.node()
        if producer is not None:
            output_producers.append(producer)
        if len(output_producers) >= len(configured_output_keys):
            break
    if len(output_producers) < len(configured_output_keys):
        warn(
            "TorchScript graph exposes fewer outputs than configured; missing output pin labels may persist "
            f"({len(output_producers)} found vs {len(configured_output_keys)} expected)."
        )
    output_attr_offset = len(configured_input_keys)
    for slot_offset, (pin_key, producer) in enumerate(zip(configured_output_keys, output_producers)):
        try:
            key = int(pin_key)
        except (TypeError, ValueError):
            continue
        if producer in node_mapping:
            # Already registered via inputs (unexpected) or duplicates; skip.
            continue
        _register_node(key, producer)
        node_mapping[producer] = key
        attrs = _bind_slot_to_node(output_attr_offset + slot_offset, key)
        attrs["pin_role"] = "output"
        attrs["pin_slot_index"] = key

    next_node_id = 0
    for node in graph_nodes:
        if node in node_mapping:
            continue
        while next_node_id in genome.nodes:
            next_node_id += 1
        _register_node(next_node_id, node)
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
    with log_timing(logger, "Loading TorchScript seed optimizers") as timing:
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
        _warn_seed_limit_violations(population, new_population)
        timing.set_details(f"assigned={len(new_population)} unique_seeds={len(unique_paths)}")

    with log_timing(logger, "Initial speciation pass") as timing:
        population.species.speciate(config, population.population, population.generation)
        timing.set_details(f"species={len(population.species.species)}")


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


def _warn_seed_limit_violations(population, genomes: dict[int, OptimizerGenome]):
    decoder = getattr(getattr(population, "guide", None), "decoder", None)
    if decoder is None:
        return
    max_nodes = getattr(decoder, "max_nodes", None)
    max_attrs = getattr(decoder, "max_attributes_per_node", None)
    try:
        max_nodes = int(max_nodes) if max_nodes is not None else None
    except (TypeError, ValueError):
        max_nodes = None
    try:
        max_attrs = int(max_attrs) if max_attrs is not None else None
    except (TypeError, ValueError):
        max_attrs = None
    if (max_nodes is None or max_nodes <= 0) and (max_attrs is None or max_attrs <= 0):
        return
    for genome in genomes.values():
        identifier = getattr(genome, "optimizer_path", None) or f"seed_genome_{genome.key}"
        if max_nodes is not None and max_nodes > 0:
            node_total = len(getattr(genome, "nodes", {}))
            if node_total > max_nodes:
                logger.warning(
                    "Seed optimizer %s defines %d nodes which exceeds decoder max_nodes=%d; guided decoder caps will truncate these graphs.",
                    identifier,
                    node_total,
                    max_nodes,
                )
        if max_attrs is not None and max_attrs > 0:
            violating = [
                ng.key
                for ng in getattr(genome, "nodes", {}).values()
                if len(getattr(ng, "dynamic_attributes", {})) > max_attrs
            ]
            if violating:
                sample = ", ".join(str(key) for key in violating[:5])
                logger.warning(
                    "Seed optimizer %s has %d node attribute sets exceeding cap (%d); example node ids: %s",
                    identifier,
                    len(violating),
                    max_attrs,
                    sample,
                )


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

    with log_timing(logger, f"Loading NEAT config ({args.config_file})") as timing:
        config = neat.Config(
            OptimizerGenome, GuidedReproduction, neat.DefaultSpeciesSet, RelativeRankStagnation, args.config_file
        )
        override_keys = tuple(GUIDED_POPULATION_FIELDS.keys())
        applied_override_keys: list[str] = []
        for key in override_keys:
            value = guided_population_overrides.get(key)
            if value is None and hasattr(config, "reproduction_config"):
                value = getattr(config.reproduction_config, key, None)
            if value is not None:
                setattr(config, key, value)
                applied_override_keys.append(key)
        test_mode_enabled = bool(getattr(args, "test_mode", False))
        if test_mode_enabled:
            setattr(config, "test_mode", True)
        max_eval_epochs: int | None = None
        if args.max_evaluation_steps is not None:
            max_eval_epochs = max(1, int(args.max_evaluation_steps))
            setattr(config, "max_evaluation_steps", max_eval_epochs)
        details = [f"pop_size={config.pop_size}"]
        if applied_override_keys:
            details.append(f"overrides={len(applied_override_keys)}")
        if test_mode_enabled:
            details.append("test_mode=1")
        if max_eval_epochs is not None:
            details.append(f"max_eval_steps={max_eval_epochs}")
        timing.set_details(", ".join(details))

    with log_timing(logger, "Constructing GuidedPopulation") as timing:
        population = GuidedPopulation(config)
        timing.set_details(f"initial_pop={len(population.population)}")

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
        try:
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
                    step = next(trainer_step_counter)
                    trainer_last_step["value"] = step
                    metrics = {
                        "trainer_generation": generation,
                        "trainer_epoch": epoch,
                        "trainer_total_epochs": total_epochs,
                        "trainer_total_loss": total_loss,
                        "trainer_kl_beta": kl_beta,
                        "trainer_learning_rate": kwargs.get("lr"),
                    }
                    metrics.update({f"trainer_loss_{name}": value for name, value in loss_terms.items()})
                    for idx, name in enumerate(available_modules):
                        metrics[f"trainer_module_active_{name}"] = 1 if name in active_modules else 0
                    clean_metrics = {k: v for k, v in metrics.items() if v is not None}
                    if clean_metrics:
                        mlflow_run.log_metrics(clean_metrics, step=step)
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
                    if active_modules:
                        log_line += f" | active_modules={','.join(active_modules)}"
                    mlflow_run.append_log_line(log_line)
                    mlflow_run.flush_log()

                population.trainer.set_progress_callback(_log_trainer_progress)

                def _log_guided_stats(stats: dict):
                    if not stats:
                        return
                    step = trainer_last_step.get("value", 0)
                    metrics = {
                        "guided_children_requested": stats.get("requested"),
                        "guided_children_created": stats.get("accepted"),
                        "guided_children_invalid_total": stats.get("invalid_total"),
                        "guided_children_repair_salvaged": stats.get("repair_salvaged"),
                        "guided_children_repair_salvaged_total": stats.get("repair_salvaged_total"),
                        "guided_latent_structure_penalty_last": stats.get("structure_penalty_last"),
                        "guided_latent_structure_penalty_mean": stats.get("structure_penalty_mean"),
                        "guided_latent_structure_penalty_samples": stats.get("structure_penalty_samples"),
                        "guided_decoder_max_nodes_hits": stats.get("decoder_max_nodes_hits"),
                        "guided_decoder_max_nodes_invalid": stats.get("decoder_max_nodes_invalid"),
                        "guided_inactive_details_total": stats.get("inactive_details_total"),
                        "guided_inactive_repair_salvaged": stats.get("inactive_repair_salvaged"),
                        "guided_inactive_repair_salvaged_total": stats.get("inactive_repair_salvaged_total"),
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
                        f"created={stats.get('accepted', 0)}, invalid_total={stats.get('invalid_total', 0)}, "
                        f"repair_salvaged={stats.get('repair_salvaged', 0)}"
                    )
                    if stats.get("decoder_max_nodes_hits"):
                        summary += f", max_nodes_hits={stats.get('decoder_max_nodes_hits')}"
                    if stats.get("inactive_details_total"):
                        summary += f", inactive_cases={stats.get('inactive_details_total')}"
                    if stats.get("inactive_repair_salvaged"):
                        summary += f", inactive_repair_salvaged={stats.get('inactive_repair_salvaged')}"
                    if stats.get("structure_penalty_samples"):
                        summary += (
                            f", structure_penalty_mean={stats.get('structure_penalty_mean', 0):.6f}"
                            f" (last={stats.get('structure_penalty_last', 0):.6f},"
                            f" samples={stats.get('structure_penalty_samples', 0)})"
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

            with log_timing(logger, f"Evolution loop ({args.num_generations} generations)") as timing:
                winner = population.run(args.num_generations)
                timing.set_details(f"completed_generations={population.generation}")
            print("\nBest genome:\n{!s}".format(winner))

            snapshot_path = None
            try:
                with log_timing(logger, "Serializing final population snapshot") as timing:
                    population_snapshot = population.snapshot_population()
                    final_dir = Path(args.final_population_dir).expanduser()
                    final_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                    filename = f"population_gen{population.generation:05d}_{timestamp}.pt"
                    snapshot_path = final_dir / filename
                    torch.save(population_snapshot, snapshot_path)
                    timing.set_details(f"path={snapshot_path}")
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
        finally:
            if mlflow_run:
                log_artifact_fn = getattr(population, "log_inactive_detail_artifact", None)
                if callable(log_artifact_fn):
                    try:
                        log_artifact_fn(mlflow_run)
                    except Exception as exc:  # pragma: no cover - logging safety net
                        logger.exception("Failed to log inactive optimizer artifact: %s", exc)
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
                    "guided_children_repair_salvaged": stats.get("repair_salvaged"),
                    "guided_children_repair_salvaged_total": stats.get("repair_salvaged_total"),
                    "guided_latent_structure_penalty_last": stats.get("structure_penalty_last"),
                    "guided_latent_structure_penalty_mean": stats.get("structure_penalty_mean"),
                    "guided_latent_structure_penalty_samples": stats.get("structure_penalty_samples"),
                    "guided_decoder_max_nodes_hits": stats.get("decoder_max_nodes_hits"),
                    "guided_decoder_max_nodes_invalid": stats.get("decoder_max_nodes_invalid"),
                    "guided_inactive_details_total": stats.get("inactive_details_total"),
                    "guided_inactive_repair_salvaged": stats.get("inactive_repair_salvaged"),
                    "guided_inactive_repair_salvaged_total": stats.get("inactive_repair_salvaged_total"),
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
                    f"created={stats.get('accepted', 0)}, invalid_total={stats.get('invalid_total', 0)}, "
                    f"repair_salvaged={stats.get('repair_salvaged', 0)}"
                )
                if stats.get("decoder_max_nodes_hits"):
                    summary += f", max_nodes_hits={stats.get('decoder_max_nodes_hits')}"
                if stats.get("inactive_details_total"):
                    summary += f", inactive_cases={stats.get('inactive_details_total')}"
                if stats.get("inactive_repair_salvaged"):
                    summary += f", inactive_repair_salvaged={stats.get('inactive_repair_salvaged')}"
                if stats.get("structure_penalty_samples"):
                    summary += (
                        f", structure_penalty_mean={stats.get('structure_penalty_mean', 0):.6f}"
                        f" (last={stats.get('structure_penalty_last', 0):.6f},"
                        f" samples={stats.get('structure_penalty_samples', 0)})"
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

        with log_timing(logger, f"Evolution loop ({args.num_generations} generations)") as timing:
            winner = population.run(args.num_generations)
            timing.set_details(f"completed_generations={population.generation}")
        print("\nBest genome:\n{!s}".format(winner))

        snapshot_path = None
        try:
            with log_timing(logger, "Serializing final population snapshot") as timing:
                population_snapshot = population.snapshot_population()
                final_dir = Path(args.final_population_dir).expanduser()
                final_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
                filename = f"population_gen{population.generation:05d}_{timestamp}.pt"
                snapshot_path = final_dir / filename
                torch.save(population_snapshot, snapshot_path)
                timing.set_details(f"path={snapshot_path}")
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
