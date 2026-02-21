import atexit
import copy
import math
import random
import re
import signal
import time
import tracemalloc
import weakref
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple
from warnings import warn

import torch
import torch.nn.functional as F
from neat.population import Population
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from genes import (
    NODE_TYPE_OPTIONS,
    NODE_TYPE_TO_INDEX,
    node_type_index_from_name,
    node_type_name_from_index,
)
from genome import OptimizerGenome
from graph_builder import *
from metrics import *
from models import *
from pareto import *
from reproduction import GuidedReproduction
from search_space_compression import *
from tasks import *

_INVALID_REASON_COUNTER: Counter = Counter()
_INVALID_REASON_REPORT_REGISTERED = False
DECODED_GRAPH_DICT_KEY = "_decoded_graph_dict"
REPAIRED_GRAPH_DICT_KEY = "_repaired_graph_dict"

PIN_ROLE_INPUT = "input"
PIN_ROLE_OUTPUT = "output"
PIN_ROLE_HIDDEN = "hidden"
_PIN_ROLE_FALLBACKS = {
    PIN_ROLE_INPUT: PIN_ROLE_INPUT,
    PIN_ROLE_OUTPUT: PIN_ROLE_OUTPUT,
    PIN_ROLE_HIDDEN: PIN_ROLE_HIDDEN,
}


def _normalize_pin_role(value: Any) -> str | None:
    if isinstance(value, str):
        role = value.strip().lower()
        if role in _PIN_ROLE_FALLBACKS:
            return _PIN_ROLE_FALLBACKS[role]
    return None


def _dump_invalid_reason_summary():
    if not _INVALID_REASON_COUNTER:
        print("[invalid-offspring] No penalized guided offspring recorded.")
        return
    total = sum(_INVALID_REASON_COUNTER.values())
    parts = ", ".join(f"{reason}={count}" for reason, count in sorted(_INVALID_REASON_COUNTER.items()))
    print(f"[invalid-offspring] total={total} :: {parts}")


def _register_invalid_reason_reporter():
    global _INVALID_REASON_REPORT_REGISTERED
    if _INVALID_REASON_REPORT_REGISTERED:
        return

    def _signal_handler(signum, frame):  # pragma: no cover - signal handler
        _dump_invalid_reason_summary()
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass
    atexit.register(_dump_invalid_reason_summary)
    _INVALID_REASON_REPORT_REGISTERED = True


class GuidedPopulation(Population):
    _optimizer_state_attr_cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
    INVALID_METRIC_VALUE = 1e6
    INVALID_PENALTY_MIN_SCALE = 0.2
    INVALID_PENALTY_DEFAULT_SCALE = 0.5
    INVALID_PENALTY_MAX_SCALE = 1.0
    MISSING_SLOT_PENALTY_MIN_SCALE = 0.35
    MISSING_SLOT_PENALTY_MAX_SCALE = 0.9
    INACTIVE_OPTIMIZER_PENALTY_SCALE = 0.7

    def __init__(self, config):
        super().__init__(config)
        _register_invalid_reason_reporter()
        graph_latent_dim = 32
        num_node_types = 10
        self.task = RegressionTask.random_init()
        self.metric_keys = self._evaluation_metric_keys(self.task)
        self.metric_best_values = self._metric_best_values(self.metric_keys)
        self.metric_guidance_weights = self._metric_guidance_weights(self.metric_keys)
        self.shared_attr_vocab = SharedAttributeVocab([], 50)
        # Keep attribute-name embeddings high-dimensional (50d) while compressing the
        # DeepSet attribute summaries down to 20d per node before DAG attention.
        attr_encoder = NodeAttributeDeepSetEncoder(
            self.shared_attr_vocab, encoder_hdim=10, aggregator_hdim=20, out_dim=20
        )
        graph_encoder = GraphEncoder(
            len(NODE_TYPE_OPTIONS),
            attr_encoder,
            latent_dim=graph_latent_dim,
            hidden_dims=[32, 32],
            pin_role_dim=2,
        )
        genome_config = getattr(config, "genome_config", None)
        configured_inputs = list(getattr(genome_config, "input_keys", [])) if genome_config else []
        configured_outputs = list(getattr(genome_config, "output_keys", [])) if genome_config else []
        output_slots: List[int] = []
        for key in configured_outputs:
            try:
                output_slots.append(int(key))
            except (TypeError, ValueError):
                continue
        max_output_slot = max(output_slots) if output_slots else -1
        decoder_min_pin_nodes = max(
            len(configured_inputs) + len(configured_outputs),
            max_output_slot + 1,
            1,
        )
        decoder = GraphDecoder(
            len(NODE_TYPE_OPTIONS),
            graph_latent_dim,
            self.shared_attr_vocab,
            graph_encoder.pin_role_embedding,
            min_pin_nodes=decoder_min_pin_nodes,
            required_input_count=len(configured_inputs),
            required_output_slots=output_slots,
        )
        icnn_hidden_dims = getattr(config, "latent_icnn_hidden_dims", (64, 32))
        if isinstance(icnn_hidden_dims, str):
            icnn_hidden_dims = [int(part) for part in icnn_hidden_dims.split(",") if part.strip()]
        if isinstance(icnn_hidden_dims, (int, float)):
            icnn_hidden_dims = [int(icnn_hidden_dims)]
        icnn_hidden_dims = tuple(int(dim) for dim in icnn_hidden_dims if int(dim) > 0)
        predictor = FitnessPredictor(
            latent_dim=graph_latent_dim,
            hidden_dim=64,
            fitness_dim=len(self.metric_keys),
            icnn_hidden_dims=icnn_hidden_dims or None,
        )

        self.guide = SelfCompressingFitnessRegularizedDAGVAE(graph_encoder, decoder, predictor)
        self.optimizer = torch.optim.Adam(self.guide.parameters(), lr=0.001)
        self.decoder_empty_penalty = float(getattr(config, "decoder_empty_penalty", 5.0))
        self.decoder_missing_node_penalty = float(getattr(config, "decoder_missing_node_penalty", 2.5))
        self.trainer = OnlineTrainer(
            self.guide,
            self.optimizer,
            metric_keys=self.metric_keys,
            decoder_empty_penalty=self.decoder_empty_penalty,
            decoder_missing_node_penalty=self.decoder_missing_node_penalty,
        )
        slice_ratio = getattr(config, "kl_partial_slice_ratio", None)
        if slice_ratio is None and hasattr(config, "reproduction_config"):
            slice_ratio = getattr(config.reproduction_config, "kl_partial_slice_ratio", None)
        if slice_ratio is not None:
            try:
                slice_ratio = float(slice_ratio)
            except (TypeError, ValueError):
                slice_ratio = None
        if slice_ratio is not None and slice_ratio <= 0:
            slice_ratio = None
        slice_dims = getattr(config, "kl_partial_slice_dims", None)
        if slice_dims is None and hasattr(config, "reproduction_config"):
            slice_dims = getattr(config.reproduction_config, "kl_partial_slice_dims", None)
        if slice_dims is not None:
            try:
                slice_dims = int(slice_dims)
            except (TypeError, ValueError):
                slice_dims = None
            else:
                if slice_dims < 0:
                    slice_dims = None
        slice_start = getattr(config, "kl_partial_slice_start", None)
        if slice_start is None and hasattr(config, "reproduction_config"):
            slice_start = getattr(config.reproduction_config, "kl_partial_slice_start", 0)
        try:
            slice_start = max(0, int(slice_start or 0))
        except (TypeError, ValueError):
            slice_start = 0
        self.trainer.kl_partial_slice_ratio = slice_ratio
        self.trainer.kl_partial_slice_dims = slice_dims
        self.trainer.kl_partial_slice_start = slice_start
        self.wl_kernel_iterations = int(getattr(config, "wl_kernel_iterations", 2))
        self.wl_kernel_loss_weight = float(getattr(config, "wl_kernel_loss_weight", 0.0))
        self.trainer.wl_kernel_iterations = self.wl_kernel_iterations
        self.trainer.wl_loss_weight = self.wl_kernel_loss_weight
        self.full_train_resize_generation = int(getattr(config, "full_train_resize_generation", 25))
        self.test_mode = bool(getattr(config, "test_mode", False))
        if self.test_mode:
            raw_scale = getattr(config, "test_epoch_scale", 0.1)
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError):
                scale = 0.1
            self.test_epoch_scale = max(0.01, min(1.0, scale))
        else:
            self.test_epoch_scale = None
        self.decoder_teacher_epochs_base = int(getattr(config, "decoder_teacher_epochs", 5))
        self.decoder_teacher_epochs_max = int(
            getattr(config, "decoder_teacher_epochs_max", max(5, self.decoder_teacher_epochs_base * 3))
        )
        self.decoder_teacher_force_weight_base = float(getattr(config, "decoder_teacher_force_weight", 2.0))
        self.decoder_teacher_force_weight_max = float(
            getattr(config, "decoder_teacher_force_weight_max", max(2.0, self.decoder_teacher_force_weight_base * 2))
        )
        self.decoder_teacher_verbose = bool(getattr(config, "decoder_teacher_verbose", True))
        self.decoder_replay_max = int(getattr(config, "decoder_replay_max", 256))
        # graph_signature_from_dict can legitimately return None (e.g., tensors with unsupported dtypes) even for validated graphs
        self._decoder_replay_cache: deque[tuple[str | None, dict]] = deque()
        self._decoder_replay_signatures: Set[str] = set()
        # reservoir keeps the most recent valid graphs to reseed future replay passes
        self._decoder_replay_reservoir: deque[tuple[str | None, dict]] = deque(maxlen=self.decoder_replay_max)
        self._decoder_replay_seeded = False
        raw_repair_seed = getattr(config, "repair_random_seed", None)
        try:
            repair_seed = int(raw_repair_seed) if raw_repair_seed is not None else None
        except (TypeError, ValueError):
            repair_seed = None
        self._repair_rng = random.Random()
        if repair_seed is not None:
            self._repair_rng.seed(repair_seed)
        self.repair_randomize_connections = bool(getattr(config, "repair_randomize_connections", False))
        self.trainer.configure_module_freeze_cycle(getattr(config, "trainer_freeze_cycle", None))
        self.trainer.module_freeze_verbose = bool(getattr(config, "trainer_freeze_verbose", False))
        self.convex_surrogate_weight = float(getattr(config, "convex_surrogate_weight", 0.5))
        beta_schedule = StagedBetaSchedule(
            start_beta=0.05,
            target_beta=0.15,
            warmup_epochs=30,
            ramp_epochs=60,
            hold_epochs=20,
            cycle_length=40,
            cycle_floor=0.02,
        )
        # This staged β-VAE schedule delays KL pressure so the decoder/aux heads learn
        # meaningful non-empty graphs before gently reintroducing structure for NEAT.
        self.trainer.configure_kl_scheduler(beta_schedule, reset_state=True)
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = GuidedReproduction(config.reproduction_config, self.reporters, stagnation)
        self.reproduction.guide_fn = self.generate_guided_offspring
        self.reproduction.optimizer_validator = lambda optimizer: self._optimizer_updates_parameters(
            optimizer, check_steps=1
        )
        self._initial_compression_done = False
        self._enable_initial_compression = getattr(config, "enable_initial_compression", False)
        self.guided_stats_callback = None
        self._guided_offspring_stats = None
        self._prev_guided_offspring_stats = None
        self.dataset_stats_callback = None
        max_evaluation_steps = getattr(config, "max_evaluation_steps", None)
        if max_evaluation_steps is not None:
            try:
                max_evaluation_steps = int(max_evaluation_steps)
            except (TypeError, ValueError):
                max_evaluation_steps = None
        if max_evaluation_steps is not None and max_evaluation_steps <= 0:
            max_evaluation_steps = None
        self.max_evaluation_steps = max_evaluation_steps
        self._last_optimizer_delta = 0.0
        self.guided_invalid_viz_enabled = bool(getattr(config, "guided_invalid_viz_enabled", True))
        raw_viz_dir = getattr(config, "guided_invalid_viz_dir", None)
        if raw_viz_dir:
            self.guided_invalid_viz_dir = Path(raw_viz_dir)
        else:
            self.guided_invalid_viz_dir = Path("debug_guided_offspring") / "invalid_graphs"
        raw_limit = getattr(config, "guided_invalid_viz_limit", 6)
        try:
            limit = int(raw_limit)
        except (TypeError, ValueError):
            limit = 6
        self.guided_invalid_viz_limit = max(0, limit)
        raw_formats = getattr(config, "guided_invalid_viz_formats", ("mermaid",))
        if isinstance(raw_formats, str):
            format_candidates = [part.strip().lower() for part in raw_formats.split(",")]
        else:
            try:
                format_candidates = [str(part).strip().lower() for part in raw_formats]
            except TypeError:
                format_candidates = []
        formats: List[str] = [fmt for fmt in format_candidates if fmt]
        if not formats:
            formats = ["mermaid"]
        # Preserve order without duplicates.
        self.guided_invalid_viz_formats = tuple(dict.fromkeys(formats))
        rankdir = str(getattr(config, "guided_invalid_viz_rankdir", "LR"))
        rankdir = rankdir.upper()
        if rankdir not in {"LR", "RL", "TB", "BT"}:
            rankdir = "LR"
        self.guided_invalid_viz_rankdir = rankdir
        self._guided_invalid_viz_generation: int | None = None
        self._guided_invalid_viz_used = 0
        self._guided_invalid_viz_import_failed = False

    def _reset_guided_offspring_stats(self):
        if self._guided_offspring_stats is not None:
            self._prev_guided_offspring_stats = self._guided_offspring_stats
        self._guided_offspring_stats = {
            "generation": self.generation,
            "requested": 0,
            "accepted": 0,
            "invalid_total": 0,
            "invalid_by_reason": Counter(),
        }

    def _accumulate_guided_offspring_stats(self, requested: int, accepted: int, invalid_counts: Counter):
        stats = self._guided_offspring_stats
        if stats is None:
            return
        stats["requested"] += int(requested or 0)
        stats["accepted"] += int(accepted or 0)
        if invalid_counts:
            total_invalid = int(sum(invalid_counts.values()))
            stats["invalid_total"] += total_invalid
            stats["invalid_by_reason"].update(invalid_counts)

    def _emit_guided_offspring_stats(self):
        stats = self._guided_offspring_stats
        if not stats:
            return
        summary = {
            "generation": stats.get("generation", self.generation),
            "requested": stats.get("requested", 0),
            "accepted": stats.get("accepted", 0),
            "invalid_total": stats.get("invalid_total", 0),
            "invalid_by_reason": dict(stats.get("invalid_by_reason", {})),
        }
        if self.guided_stats_callback:
            self.guided_stats_callback(summary)
        if summary["invalid_total"]:
            parts = ", ".join(f"{reason}={count}" for reason, count in sorted(summary["invalid_by_reason"].items()))
            self.reporters.info(
                f"Post-repair guided offspring invalid counts: total={summary['invalid_total']} :: {parts}"
            )
        self._guided_offspring_stats = summary

    def _emit_dataset_stats(self, valid_size: int, invalid_size: int):
        if not self.dataset_stats_callback:
            return
        summary = {
            "generation": self.generation,
            "valid": int(valid_size or 0),
            "invalid": int(invalid_size or 0),
        }
        summary["total"] = summary["valid"] + summary["invalid"]
        self.dataset_stats_callback(summary)

    def _guided_empty_ratio(self) -> float:
        stats = self._prev_guided_offspring_stats or {}
        requested = float(stats.get("requested", 0) or 0)
        invalid = stats.get("invalid_by_reason", {}) or {}
        empties = float(invalid.get("empty_graph", 0) or 0)
        if requested <= 0:
            return 0.0
        return max(0.0, min(1.0, empties / requested))

    def _decoder_refresh_schedule(self) -> tuple[int, float]:
        ratio = self._guided_empty_ratio()
        epochs_span = max(0, self.decoder_teacher_epochs_max - self.decoder_teacher_epochs_base)
        weight_span = max(0.0, self.decoder_teacher_force_weight_max - self.decoder_teacher_force_weight_base)
        epochs = int(round(self.decoder_teacher_epochs_base + epochs_span * ratio))
        weight = float(self.decoder_teacher_force_weight_base + weight_span * ratio)
        epochs = max(self.decoder_teacher_epochs_base, min(self.decoder_teacher_epochs_max, epochs))
        weight = max(self.decoder_teacher_force_weight_base, min(self.decoder_teacher_force_weight_max, weight))
        return epochs, weight

    def _trainer_epoch_schedule(self) -> Dict[str, float]:
        """Return the epoch/warmup schedule for the current generation."""
        if self.generation == 0:
            schedule = {
                "epochs": 100,
                "warmup_epochs": 10,
                "loss_threshold": 0.97,
                "baseline_window": 5,
            }
        elif self.generation < self.full_train_resize_generation:
            schedule = {
                "epochs": 50,
                "warmup_epochs": 5,
                "loss_threshold": 0.98,
                "baseline_window": 5,
            }
        else:
            schedule = {
                "epochs": 10,
                "warmup_epochs": 3,
                "loss_threshold": 0.99,
                "baseline_window": 3,
            }
        return self._apply_test_epoch_reduction(schedule)

    def _apply_test_epoch_reduction(self, schedule: Dict[str, float]) -> Dict[str, float]:
        if not self.test_mode:
            return schedule
        scaled = dict(schedule)
        scale = self.test_epoch_scale or 0.1
        scaled_epochs = max(1, int(math.ceil(schedule["epochs"] * scale)))
        scaled["epochs"] = scaled_epochs
        scaled["warmup_epochs"] = min(
            scaled_epochs,
            max(1, int(math.ceil(schedule["warmup_epochs"] * scale))),
        )
        scaled["baseline_window"] = min(
            scaled_epochs,
            max(1, int(math.ceil(schedule["baseline_window"] * scale))),
        )
        scaled["loss_threshold"] = min(schedule["loss_threshold"], 0.9)
        return scaled

    def _buffer_decoder_replay_dict(self, graph_dict: dict | None):
        if not graph_dict or self.decoder_replay_max <= 0:
            return
        cloned = self._clone_graph_dict(graph_dict)
        if cloned is None:
            return
        signature = graph_signature_from_dict(cloned)
        if signature and signature in self._decoder_replay_signatures:
            return
        if self.decoder_replay_max > 0 and len(self._decoder_replay_cache) >= self.decoder_replay_max:
            old_signature, _ = self._decoder_replay_cache.popleft()
            if old_signature:
                self._decoder_replay_signatures.discard(old_signature)
        self._decoder_replay_cache.append((signature, cloned))
        if signature:
            self._decoder_replay_signatures.add(signature)
        if self.decoder_replay_max > 0:
            # reservoir keeps the most recent valid graphs to reseed future replay passes
            if len(self._decoder_replay_reservoir) >= self.decoder_replay_max:
                self._decoder_replay_reservoir.popleft()
            self._decoder_replay_reservoir.append((signature, cloned))

    def _seed_decoder_replay_from_population(self) -> int:
        """Populate decoder replay buffers with the current population graphs once."""
        if self._decoder_replay_seeded or self.decoder_replay_max <= 0:
            return 0
        seeded = 0
        for genome in self.population.values():
            graph_dict = getattr(genome, "graph_dict", None)
            if graph_dict is None:
                self.genome_to_data(genome)
                graph_dict = getattr(genome, "graph_dict", None)
            if not graph_dict:
                continue
            self._buffer_decoder_replay_dict(graph_dict)
            seeded += 1
        if seeded:
            self.reporters.info(f"Seeded decoder replay with {seeded} initial population graphs")
        self._decoder_replay_seeded = True
        return seeded

    def _record_decoder_failure(
        self,
        graph_dict: dict | None,
        fitnesses: dict,
        reason: str = "empty_graph",
    ) -> None:
        if not graph_dict:
            return
        data = self.trainer._graph_dict_to_data(graph_dict)
        if data is None:
            return
        try:
            self.trainer.add_data(
                [data],
                [fitnesses],
                invalid_flags=[True],
            )
        except Exception as exc:
            warn(f"Failed to record decoder failure ({reason}): {exc}")

    def _maybe_visualize_guided_invalid_graph(
        self,
        genome: OptimizerGenome,
        graph_dict: dict | None,
        *,
        reason: str | None,
        child_index: int | None = None,
        num_edges: int | None = None,
    ) -> None:
        if not self.guided_invalid_viz_enabled or not graph_dict:
            return
        if self.guided_invalid_viz_limit <= 0:
            return
        generation = getattr(self, "generation", None)
        if generation != self._guided_invalid_viz_generation:
            self._guided_invalid_viz_generation = generation
            self._guided_invalid_viz_used = 0
        if self._guided_invalid_viz_used >= self.guided_invalid_viz_limit:
            return
        stored_repaired = graph_dict.get(REPAIRED_GRAPH_DICT_KEY)
        if stored_repaired is not None:
            repaired_graph = self._clone_graph_dict(stored_repaired, include_history=False)
        else:
            repaired_graph = self._clone_graph_dict(graph_dict, include_history=False)
        if repaired_graph is None:
            return
        decoded_graph = None
        if DECODED_GRAPH_DICT_KEY in graph_dict:
            base_graph = graph_dict[DECODED_GRAPH_DICT_KEY]
            decoded_graph = self._clone_graph_dict(base_graph, include_history=False)
        graph_for_entry = decoded_graph or repaired_graph
        try:
            self.guided_invalid_viz_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            warn(f"Failed to create invalid-graph visualization dir '{self.guided_invalid_viz_dir}': {exc}")
            self.guided_invalid_viz_enabled = False
            return
        try:
            from population_visualizer import RenderContext, build_mermaid_graph
        except Exception as exc:  # pragma: no cover - import guard
            if not self._guided_invalid_viz_import_failed:
                warn(f"Unable to import population_visualizer for invalid graph rendering: {exc}")
                self._guided_invalid_viz_import_failed = True
            return
        label_reason = reason or "invalid_graph"
        sanitized_reason = re.sub(r"[^A-Za-z0-9_.-]+", "_", label_reason)
        gen_label = generation if isinstance(generation, int) else "X"
        child_label = child_index if child_index is not None else "X"
        parts = [f"gen{gen_label}", f"child{child_label}"]
        if num_edges is not None:
            parts.append(f"edges{num_edges}")
        parts.append(sanitized_reason)
        base_name = "_".join(str(part) for part in parts if part != "")
        entry = {
            "genome_id": getattr(genome, "key", None),
            "species_id": getattr(genome, "species_id", None),
            "fitness": getattr(genome, "fitness", None),
            "invalid_graph": True,
            "invalid_reason": label_reason,
            "graph": graph_for_entry,
        }
        context = RenderContext(
            generation=generation if isinstance(generation, int) else None,
            task=getattr(self.task, "name", None),
        )
        produced_any = False

        supported_format = False
        for fmt in self.guided_invalid_viz_formats:
            if fmt == "mermaid":
                supported_format = True
            else:
                warn(f"Unsupported guided invalid visualization format '{fmt}'; only 'mermaid' is available.")
        if not supported_format:
            return

        variants: List[Tuple[str, dict]] = []
        if decoded_graph is not None:
            variants.append(("decoded", decoded_graph))
        variants.append(("repaired", repaired_graph))

        for suffix, graph_payload in variants:
            mermaid_entry = dict(entry)
            mermaid_entry["graph"] = graph_payload
            try:
                mermaid_source = build_mermaid_graph(
                    mermaid_entry,
                    context=context,
                    rankdir=self.guided_invalid_viz_rankdir,
                    highlight_invalid=True,
                )
                mermaid_path = self.guided_invalid_viz_dir / f"{base_name}_{suffix}.mmd"
                mermaid_path.write_text(mermaid_source)
                produced_any = True
            except Exception as exc:
                warn(f"Failed to write invalid graph Mermaid file '{base_name}_{suffix}': {exc}")

        if produced_any:
            self._guided_invalid_viz_used += 1

    def _consume_decoder_replay_graphs(self) -> list[dict]:
        if not self._decoder_replay_cache:
            if not self._decoder_replay_reservoir:
                return []
            reseed = list(self._decoder_replay_reservoir)
            self._decoder_replay_cache.extend(reseed)
            for signature, _ in reseed:
                if signature:
                    self._decoder_replay_signatures.add(signature)
        payload = [entry[1] for entry in self._decoder_replay_cache]
        self._decoder_replay_cache.clear()
        self._decoder_replay_signatures.clear()
        return payload

    @staticmethod
    def _evaluation_metric_keys(task) -> List[Metric]:
        metrics: List[Metric] = list(task.metrics)
        metrics.extend([AreaUnderTaskMetrics, TimeCost, MemoryCost])
        return sort_metrics_by_name(metrics)

    @staticmethod
    def _metric_best_values(metrics: List[Metric]) -> List[float]:
        return [metric_best_value(metric) for metric in metrics]

    @staticmethod
    def _metric_guidance_weights(metrics: List[Metric]) -> List[float]:
        return [float(getattr(metric, "guidance_weight", 1.0)) for metric in metrics]

    def genome_to_data(self, genome: OptimizerGenome):
        # always rebuild graph_dict so that new attributes are captured
        # but preserve any richer metadata (graph_ir/module_state) from prior exports
        existing_graph = getattr(genome, "graph_dict", None) or {}
        preserved_ir = copy.deepcopy(existing_graph.get("graph_ir")) if existing_graph else None
        preserved_state = copy.deepcopy(existing_graph.get("module_state")) if existing_graph else None
        preserved_type = existing_graph.get("module_type") if existing_graph else None

        # sort by node id so positions line up
        node_ids = sorted(genome.nodes.keys())
        node_types = []
        node_attributes = []
        raw_input_keys = [key for key in getattr(self.config.genome_config, "input_keys", [])]
        raw_output_keys = [key for key in getattr(self.config.genome_config, "output_keys", [])]
        input_keys = {
            int(key)
            for key in raw_input_keys
            if isinstance(key, (int, float)) or (isinstance(key, str) and key.strip())
        }
        output_keys = {
            int(key)
            for key in raw_output_keys
            if isinstance(key, (int, float)) or (isinstance(key, str) and key.strip())
        }
        input_slot_ranks = {}
        for order, key in enumerate(raw_input_keys):
            try:
                input_slot_ranks[int(key)] = order
            except (TypeError, ValueError):
                continue
        output_slot_ranks = {}
        for order, key in enumerate(raw_output_keys):
            try:
                output_slot_ranks[int(key)] = order
            except (TypeError, ValueError):
                continue

        def _role_for_node_key(node_key: Any) -> str:
            try:
                key = int(node_key)
            except (TypeError, ValueError):
                return PIN_ROLE_HIDDEN
            if key in input_keys:
                return PIN_ROLE_INPUT
            if key in output_keys:
                return PIN_ROLE_OUTPUT
            return PIN_ROLE_HIDDEN

        for nid in node_ids:
            node = genome.nodes[nid]
            idx = NODE_TYPE_TO_INDEX.get(node.node_type)
            if idx is None:
                raise KeyError(f"Unknown node_type {node.node_type!r}")
            attr_names = [attribute_key_to_name(a) for a in node.dynamic_attributes.keys()]
            attr_dict = copy.deepcopy(node.dynamic_attributes)
            role = _role_for_node_key(nid)
            attr_dict["pin_role"] = role
            attr_names.append("pin_role")
            attr_dict["is_input_pin"] = 1.0 if role == PIN_ROLE_INPUT else 0.0
            attr_dict["is_output_pin"] = 1.0 if role == PIN_ROLE_OUTPUT else 0.0
            attr_names.extend(["is_input_pin", "is_output_pin"])
            slot_rank = None
            try:
                node_key_int = int(nid)
            except (TypeError, ValueError):
                node_key_int = None
            if node_key_int is not None:
                if role == PIN_ROLE_INPUT:
                    slot_rank = input_slot_ranks.get(node_key_int)
                elif role == PIN_ROLE_OUTPUT:
                    slot_rank = output_slot_ranks.get(node_key_int)
            if slot_rank is not None:
                attr_dict["pin_slot_index"] = int(slot_rank)
                attr_names.append("pin_slot_index")
            self.shared_attr_vocab.add_names(attr_names)
            node_types.append(idx)
            node_attributes.append(attr_dict)
        node_types = torch.tensor(node_types, dtype=torch.long)

        edges = []
        for (src, dst), conn in genome.connections.items():
            if conn.enabled:
                if src in node_ids and dst in node_ids:
                    local_src = node_ids.index(src)
                    local_dst = node_ids.index(dst)
                    edges.append([local_src, local_dst])
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_index_copy = edge_index.clone() if edge_index.numel() > 0 else edge_index
        graph_dict = {
            "node_types": node_types.clone(),
            "edge_index": edge_index_copy,
            "node_attributes": copy.deepcopy(node_attributes),
        }
        slot_shapes = getattr(genome, "slot_shapes", None)
        if slot_shapes is not None:
            graph_dict["slot_shapes"] = copy.deepcopy(slot_shapes)
        if getattr(genome, "serialized_module", None) is not None:
            graph_dict["serialized_module"] = genome.serialized_module
        if preserved_ir is not None:
            graph_dict["graph_ir"] = preserved_ir
        if preserved_state is not None:
            graph_dict["module_state"] = preserved_state
        if preserved_type is not None:
            graph_dict["module_type"] = preserved_type
        genome.graph_dict = graph_dict
        return Data(node_types=node_types, edge_index=edge_index, node_attributes=node_attributes)

    @staticmethod
    def _clone_graph_dict(graph_dict: dict | None, *, include_history: bool = True) -> dict | None:
        if not graph_dict:
            return None
        cloned = {}
        node_types = graph_dict.get("node_types")
        if node_types is not None:
            cloned["node_types"] = node_types.clone().detach().cpu()
        edge_index = graph_dict.get("edge_index")
        if edge_index is not None:
            cloned["edge_index"] = edge_index.clone().detach().cpu()
        if include_history:
            decoded_graph = graph_dict.get(DECODED_GRAPH_DICT_KEY)
            if decoded_graph is not None:
                cloned[DECODED_GRAPH_DICT_KEY] = copy.deepcopy(decoded_graph)
            repaired_graph = graph_dict.get(REPAIRED_GRAPH_DICT_KEY)
            if repaired_graph is not None:
                cloned[REPAIRED_GRAPH_DICT_KEY] = copy.deepcopy(repaired_graph)
        attributes = []
        for attr in graph_dict.get("node_attributes", []) or []:
            cloned_attr = {}
            for key, value in attr.items():
                if torch.is_tensor(value):
                    cloned_attr[key] = value.clone().detach().cpu()
                else:
                    cloned_attr[key] = copy.deepcopy(value)
            attributes.append(cloned_attr)
        cloned["node_attributes"] = attributes
        slot_shapes = graph_dict.get("slot_shapes")
        if slot_shapes is not None:
            cloned["slot_shapes"] = copy.deepcopy(slot_shapes)
        serialized = graph_dict.get("serialized_module")
        if serialized is not None:
            cloned["serialized_module"] = bytes(serialized)
        graph_ir = graph_dict.get("graph_ir")
        if graph_ir is not None:
            cloned["graph_ir"] = copy.deepcopy(graph_ir)
        module_state = graph_dict.get("module_state")
        if module_state is not None:
            cloned["module_state"] = copy.deepcopy(module_state)
        module_type = graph_dict.get("module_type")
        if module_type is not None:
            cloned["module_type"] = module_type
        return cloned

    @staticmethod
    def _graph_dict_to_data(graph_dict: dict | None) -> Data | None:
        if not graph_dict:
            return None
        node_types = graph_dict.get("node_types")
        edge_index = graph_dict.get("edge_index")
        node_attributes = graph_dict.get("node_attributes")
        if node_types is None or edge_index is None or node_attributes is None:
            return None
        node_types = node_types.clone().detach().long()
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.clone().detach().long()
        else:
            edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        data = Data(node_types=node_types, edge_index=edge_index, node_attributes=node_attributes)
        return data

    def _minimum_required_node_count(self) -> int:
        config = getattr(self.config, "genome_config", None)
        if config is None:
            return 1
        input_keys = list(getattr(config, "input_keys", []))
        output_keys = list(getattr(config, "output_keys", []))
        required_inputs = len(input_keys)
        output_slots: List[int] = []
        for key in output_keys:
            try:
                output_slots.append(int(key))
            except (TypeError, ValueError):
                continue
        max_output_slot = max(output_slots) if output_slots else -1
        slot_bound = max_output_slot + 1 if max_output_slot >= 0 else 0
        return max(required_inputs, len(output_keys), slot_bound, 1)

    @staticmethod
    def _ensure_graph_node_capacity(graph_dict: dict, min_nodes: int) -> int:
        if not graph_dict or min_nodes <= 0:
            return 0
        node_attrs = graph_dict.get("node_attributes") or []
        if not isinstance(node_attrs, list):
            node_attrs = list(node_attrs)
        original_len = len(node_attrs)
        while len(node_attrs) < min_nodes:
            node_attrs.append({})
        graph_dict["node_attributes"] = node_attrs

        desired = len(node_attrs)
        node_types = graph_dict.get("node_types")
        if isinstance(node_types, torch.Tensor):
            if node_types.numel() < desired:
                pad = torch.zeros(desired - node_types.numel(), dtype=node_types.dtype)
                graph_dict["node_types"] = torch.cat([node_types, pad], dim=0)
        elif isinstance(node_types, list):
            while len(node_types) < desired:
                node_types.append(0)
            graph_dict["node_types"] = node_types
        elif desired > 0:
            graph_dict["node_types"] = torch.zeros(desired, dtype=torch.long)

        edge_index = graph_dict.get("edge_index")
        if edge_index is None:
            graph_dict["edge_index"] = torch.empty((2, 0), dtype=torch.long)

        return len(node_attrs) - original_len

    def _prepare_decoded_graph_dict(self, graph_dict: dict | None) -> Tuple[int, List[Tuple[int, int]]]:
        if not graph_dict:
            return 0, []
        node_types = graph_dict.get("node_types")
        if node_types is None:
            node_count = 0
        elif isinstance(node_types, torch.Tensor):
            node_count = int(node_types.numel())
        else:
            node_count = len(node_types)

        if node_count <= 0:
            graph_dict["edge_index"] = torch.empty((2, 0), dtype=torch.long)
            edges: List[Tuple[int, int]] = []
        else:
            edges = self._edge_list_from_index(graph_dict.get("edge_index"), node_count)
            if edges:
                tensor = torch.as_tensor(edges, dtype=torch.long).t().contiguous()
            else:
                tensor = torch.empty((2, 0), dtype=torch.long)
            graph_dict["edge_index"] = tensor

        if DECODED_GRAPH_DICT_KEY not in graph_dict:
            cloned = self._clone_graph_dict(graph_dict, include_history=False)
            if cloned is not None:
                graph_dict[DECODED_GRAPH_DICT_KEY] = cloned
        return node_count, edges

    @staticmethod
    def _fitness_value(genome) -> float:
        value = getattr(genome, "fitness", None)
        if value is None:
            return float("-inf")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    def snapshot_population(self) -> dict:
        """Capture a serializable snapshot of the current population state."""
        species_dict = getattr(self.species, "species", {}) or {}
        species_assignments: Dict[int, int] = {}
        for sid, species in species_dict.items():
            members = getattr(species, "members", {}) or {}
            if isinstance(members, dict):
                member_ids = members.keys()
            else:
                member_ids = members
            for gid in member_ids:
                try:
                    species_assignments[int(gid)] = sid
                except (TypeError, ValueError):
                    continue

        entries = []
        for gid in sorted(self.population.keys()):
            genome = self.population[gid]
            if getattr(genome, "graph_dict", None) is None:
                try:
                    self.genome_to_data(genome)
                except Exception:
                    pass
            graph_dict = self._clone_graph_dict(getattr(genome, "graph_dict", None))
            entry = {
                "genome_id": gid,
                "species_id": species_assignments.get(gid),
                "fitness": getattr(genome, "fitness", None),
                "fitnesses": copy.deepcopy(getattr(genome, "fitnesses", None)),
                "invalid_graph": getattr(genome, "invalid_graph", False),
                "invalid_reason": getattr(genome, "invalid_reason", None),
                "optimizer_path": getattr(genome, "optimizer_path", None),
                "graph": graph_dict,
            }
            entries.append(entry)

        task_name = None
        name_attr = getattr(self.task, "name", None)
        if callable(name_attr):
            try:
                task_name = name_attr()
            except Exception:
                task_name = None

        snapshot = {
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
            "generation": int(getattr(self, "generation", -1)),
            "population_size": len(self.population),
            "species_count": len(species_dict),
            "task": task_name,
            "entries": entries,
        }
        return snapshot

    def generate_guided_offspring(
        self,
        starting_genomes: Sequence[OptimizerGenome],
        config,
        n_offspring: int = 10,
        latent_steps: int = 50,
        latent_lr: float = 1e-2,
        max_decode_attempts: int = 5,
        decode_jitter_std: float = 0.05,
        latent_tether_weight: float | None = None,
        latent_tether_init: float = 1e-3,
        latent_tether_prior: float = 1e-4,
    ) -> List[OptimizerGenome]:
        """
        For a fixed (task_type, task_features), optimize `z_g` in latent space to maximize
        the surrogate predictor, decode each optimized z_g back into a DAG, then
        convert those DAGs into new NEAT genomes.
        """
        empty_feedback = self._guided_empty_ratio()
        latent_steps = max(5, int(latent_steps * (1.0 - 0.5 * empty_feedback)))
        latent_lr = float(latent_lr) * (0.5 + 0.5 * (1.0 - empty_feedback))
        latent_tether_init = float(latent_tether_init) * (1.0 + empty_feedback)
        _ = decode_jitter_std  # legacy arg retained; repair hook replaces jitter retries

        metric_keys = self.metric_keys
        metric_best_values = self.metric_best_values
        metric_guidance_weights = self.metric_guidance_weights
        metric_dim = len(metric_keys)

        # 2) Initialize random graph latents and set requires_grad=True
        graph_latent_dim = self.guide.graph_encoder.latent_dim
        # encode the optimizer graphs from top half of starting_genomes then optimize, rest are cross-species
        sorted_genomes = sorted(starting_genomes, key=lambda g: g.fitness, reverse=True)
        num_encode = min(max(n_offspring // 2, 0), len(sorted_genomes))
        top_genomes = sorted_genomes[:num_encode]
        data_list = []
        for g in top_genomes:
            graph_dict = getattr(g, "graph_dict", None)
            if graph_dict is None:
                # Lazily build the cached graph if this genome has never been encoded.
                self.genome_to_data(g)
                graph_dict = g.graph_dict
            data_list.append(
                Data(
                    node_types=graph_dict["node_types"].clone().detach().long(),
                    edge_index=graph_dict["edge_index"].clone().detach().long(),
                    node_attributes=graph_dict["node_attributes"],
                )
            )

        if data_list:
            batch = Batch.from_data_list(data_list)
            mu_g, lv_g = self.guide.graph_encoder(
                batch.node_types,
                batch.edge_index,
                batch.node_attributes,
                batch.batch,
                num_graphs=batch.num_graphs,
            )
            z_g_encoded = self.guide.reparameterize(mu_g, lv_g, self.guide.graph_latent_mask)
            z_g_encoded = z_g_encoded.clone().detach().requires_grad_(True)
        else:
            z_g_encoded = torch.empty(
                (0, graph_latent_dim), device=self.guide.graph_latent_mask.device, dtype=torch.float32
            )
        num_random = n_offspring - z_g_encoded.size(0)
        if num_random > 0:
            z_g_random = torch.randn((num_random, graph_latent_dim), device=z_g_encoded.device, requires_grad=True)
            z_g = torch.cat([z_g_encoded, z_g_random], dim=0).clone().detach().requires_grad_(True)
        else:
            z_g = z_g_encoded

        z_g_initial = z_g.detach().clone()
        learnable_tether = latent_tether_weight is None

        total_latents = z_g.size(0)
        if total_latents == 0:
            return []

        predictor_dim = self.guide.fitness_predictor.output_dim

        def pad_metric_tensor(values: Sequence[float], fill_value: float) -> torch.Tensor:
            tensor = torch.as_tensor(values, dtype=z_g.dtype, device=z_g.device)
            if tensor.numel() < predictor_dim:
                tensor = F.pad(tensor, (0, predictor_dim - tensor.numel()), value=fill_value)
            elif tensor.numel() > predictor_dim:
                tensor = tensor[:predictor_dim]
            return tensor

        best_tensor = pad_metric_tensor(metric_best_values, 0.0)
        weight_tensor = pad_metric_tensor(metric_guidance_weights, 1.0)
        if best_tensor.dim() == 1:
            best_tensor = best_tensor.unsqueeze(0)
        if weight_tensor.dim() == 1:
            weight_tensor = weight_tensor.unsqueeze(0)
        best_tensor = best_tensor.expand(total_latents, -1)
        weight_tensor = weight_tensor.expand(total_latents, -1)

        tether_params: List[torch.Tensor] = []
        latent_tether_logit = None
        if learnable_tether:
            if latent_tether_init <= 0:
                latent_tether_init = 1e-6
            init_value = torch.full((total_latents, 1), float(latent_tether_init), device=z_g.device)
            # inverse softplus to match the requested initial value.
            latent_tether_logit = torch.log(torch.expm1(init_value)).clamp(min=-20.0, max=20.0)
            latent_tether_logit = latent_tether_logit.detach().clone().requires_grad_(True)
            tether_params.append(latent_tether_logit)

        opt_params: List[torch.Tensor] = [z_g]
        opt_params.extend(tether_params)
        opt = torch.optim.Adam(opt_params, lr=latent_lr)
        for _ in range(latent_steps):
            pred, _, convex_pred = self.guide.fitness_predictor(z_g)
            guiding_pred = convex_pred if convex_pred is not None else pred
            canonical = canonical_log_distance(guiding_pred, best_tensor)
            weighted = canonical.pow(2) * weight_tensor
            loss = weighted.sum(dim=1).mean()
            per_latent_tether = F.mse_loss(z_g, z_g_initial, reduction="none").mean(dim=1)
            if learnable_tether and latent_tether_logit is not None:
                weights = F.softplus(latent_tether_logit).view(-1)
                tether = (weights * per_latent_tether).mean()
                loss = loss + tether
                if latent_tether_prior > 0:
                    target = torch.full_like(weights, float(latent_tether_init))
                    prior = F.mse_loss(weights, target)
                    loss = loss + latent_tether_prior * prior
            elif latent_tether_weight:
                loss = loss + float(latent_tether_weight) * per_latent_tether.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        stats = []
        new_genomes = []
        empty_graph_count = 0
        duplicate_count = 0
        rebuild_failures = 0
        inactive_optimizer_count = 0
        missing_slot_rejections = 0
        repair_failures = 0
        debug_dir = Path("debug_guided_offspring")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_save_limit = 5
        debug_saved = 0
        generation_idx = getattr(self, "generation", -1)
        seen_graph_signatures: Set[str] = set()
        total_requested = z_g.size(0)
        invalid_reason_counts: Counter = Counter()
        last_valid_graph_dict = None

        for i in range(total_requested):
            latent = z_g[i].unsqueeze(0).clone()
            with torch.no_grad():
                decoded = self.guide.decode(latent)
            if isinstance(decoded, tuple):
                decoded_graphs, _ = decoded
            else:
                decoded_graphs = decoded
            graph_dict = decoded_graphs[0]
            min_nodes_required = self._minimum_required_node_count()
            original_node_count = len(graph_dict.get("node_attributes") or [])
            self._ensure_graph_node_capacity(graph_dict, min_nodes_required)
            if original_node_count < min_nodes_required:
                reason = "insufficient_node_count"
                genome = genome_from_graph_dict(graph_dict, self.config.genome_config, key=i)
                num_edges = len(genome.connections)
                self._maybe_visualize_guided_invalid_graph(
                    genome,
                    graph_dict,
                    reason=reason,
                    child_index=i,
                    num_edges=num_edges,
                )
                genome.graph_dict = graph_dict
                penalty_details = {
                    "required_nodes": min_nodes_required,
                    "decoded_nodes": original_node_count,
                }
                self._assign_penalty(
                    genome,
                    reason=reason,
                    skip_evaluation=True,
                    penalty_details=penalty_details,
                    graph_dict=graph_dict,
                    record_decoder_failure=True,
                )
                self._buffer_decoder_replay_dict(graph_dict)
                if last_valid_graph_dict is not None:
                    self._buffer_decoder_replay_dict(last_valid_graph_dict)
                invalid_reason_counts[reason] += 1
                graph_signature = graph_signature_from_dict(graph_dict)
                if graph_signature:
                    seen_graph_signatures.add(graph_signature)
                new_genomes.append(genome)
                continue
            self._prepare_decoded_graph_dict(graph_dict)
            repaired = self._repair_graph_dict(graph_dict)
            repair_reason = graph_dict.get("_repair_failure_reason")
            repair_details = graph_dict.get("_repair_failure_details")
            if repaired is False:
                repair_failures += 1
            genome = genome_from_graph_dict(graph_dict, self.config.genome_config, key=i)
            num_edges = len(genome.connections)
            num_params = len(genome.connections)
            graph_signature = graph_signature_from_dict(graph_dict)

            if repair_reason in {"missing_input_slots", "missing_output_slots"}:
                self._maybe_visualize_guided_invalid_graph(
                    genome,
                    graph_dict,
                    reason=repair_reason,
                    child_index=i,
                    num_edges=num_edges,
                )
                genome.graph_dict = graph_dict
                penalty_details = repair_details or {}
                self._assign_penalty(
                    genome,
                    reason=repair_reason,
                    skip_evaluation=True,
                    penalty_details=penalty_details,
                    graph_dict=graph_dict,
                    record_decoder_failure=True,
                )
                self._buffer_decoder_replay_dict(graph_dict)
                if last_valid_graph_dict is not None:
                    self._buffer_decoder_replay_dict(last_valid_graph_dict)
                invalid_reason_counts[repair_reason] += 1
                if graph_signature:
                    seen_graph_signatures.add(graph_signature)
                new_genomes.append(genome)
                continue

            if num_edges == 0 or debug_saved < debug_save_limit:
                debug_path = debug_dir / f"gen{generation_idx}_child{i}_edges{num_edges}.pt"
                torch.save(graph_dict, debug_path)
                debug_saved += 1

            if num_edges == 0:
                empty_graph_count += 1
                warn("Guided offspring decoder produced an empty graph (no edges); assigning penalty fitness.")
                self._maybe_visualize_guided_invalid_graph(
                    genome,
                    graph_dict,
                    reason="empty_graph",
                    child_index=i,
                    num_edges=num_edges,
                )
                genome.graph_dict = graph_dict
                penalty_metrics = self._assign_penalty(
                    genome,
                    reason="empty_graph",
                    skip_evaluation=True,
                    graph_dict=graph_dict,
                    record_decoder_failure=True,
                )
                self._buffer_decoder_replay_dict(graph_dict)
                if last_valid_graph_dict is not None:
                    self._buffer_decoder_replay_dict(last_valid_graph_dict)
                invalid_reason_counts["empty_graph"] += 1
                if graph_signature:
                    seen_graph_signatures.add(graph_signature)
                new_genomes.append(genome)
                continue

            if graph_signature in seen_graph_signatures:
                duplicate_count += 1
                warn("Guided offspring decoder produced a duplicate graph; skipping to preserve diversity.")
                self._buffer_decoder_replay_dict(graph_dict)
                if last_valid_graph_dict is not None:
                    self._buffer_decoder_replay_dict(last_valid_graph_dict)
                continue

            slot_ok, slot_details = self._graph_output_slot_coverage(graph_dict)
            if not slot_ok:
                missing_slot_rejections += 1
                detail_parts = []
                missing_slots = slot_details.get("missing_slots", [])
                slot_attrs = slot_details.get("attributes", {})
                for slot_idx in missing_slots:
                    attr_names = slot_attrs.get(slot_idx) or []
                    attrs_summary = ",".join(attr_names) if attr_names else "<none>"
                    detail_parts.append(f"slot{slot_idx}[attrs={attrs_summary}]")
                message = "Guided offspring graph never routed into optimizer outputs; " + (
                    "; ".join(detail_parts) if detail_parts else "no slot metadata available"
                )
                warn(message + "; assigning penalty fitness.")
                self._maybe_visualize_guided_invalid_graph(
                    genome,
                    graph_dict,
                    reason="missing_output_slots",
                    child_index=i,
                    num_edges=num_edges,
                )
                genome.graph_dict = graph_dict
                total_slots = len(getattr(self.config.genome_config, "output_keys", []))
                wrong_type_slots = slot_details.get("wrong_type_slots") or slot_details.get("wrong_role_slots") or []
                penalty_details = {
                    "missing_slots": missing_slots,
                    "wrong_type_slots": wrong_type_slots,
                    "total_slots": total_slots,
                    "missing_count": len(set(missing_slots) | set(wrong_type_slots)),
                }
                self._assign_penalty(
                    genome,
                    reason="missing_output_slots",
                    skip_evaluation=True,
                    penalty_details=penalty_details,
                    graph_dict=graph_dict,
                    record_decoder_failure=True,
                )
                self._buffer_decoder_replay_dict(graph_dict)
                if last_valid_graph_dict is not None:
                    self._buffer_decoder_replay_dict(last_valid_graph_dict)
                invalid_reason_counts["missing_output_slots"] += 1
                if graph_signature:
                    seen_graph_signatures.add(graph_signature)
                new_genomes.append(genome)
                continue

            optimizer = rebuild_and_script(graph_dict, self.config.genome_config, key=i, genome=genome)
            if optimizer:
                if not self._optimizer_updates_parameters(optimizer, check_steps=2):
                    inactive_optimizer_count += 1
                    warn("Guided offspring optimizer failed to modify model parameters; assigning penalty fitness.")
                    self._maybe_visualize_guided_invalid_graph(
                        genome,
                        graph_dict,
                        reason="inactive_optimizer",
                        child_index=i,
                        num_edges=num_edges,
                    )
                    genome.graph_dict = graph_dict
                    penalty_details = {"parameter_delta": getattr(self, "_last_optimizer_delta", 0.0)}
                    self._assign_penalty(
                        genome,
                        reason="inactive_optimizer",
                        skip_evaluation=True,
                        penalty_details=penalty_details,
                        graph_dict=graph_dict,
                        record_decoder_failure=True,
                    )
                    self._buffer_decoder_replay_dict(graph_dict)
                    if last_valid_graph_dict is not None:
                        self._buffer_decoder_replay_dict(last_valid_graph_dict)
                    invalid_reason_counts["inactive_optimizer"] += 1
                    if graph_signature:
                        seen_graph_signatures.add(graph_signature)
                    new_genomes.append(genome)
                    continue

                genome.optimizer = optimizer
                genome.graph_dict = graph_dict
                last_valid_graph_dict = self._clone_graph_dict(graph_dict)
                self._buffer_decoder_replay_dict(last_valid_graph_dict)
                if graph_signature:
                    seen_graph_signatures.add(graph_signature)
                new_genomes.append(genome)
                stats.append((i, num_edges, num_params))
                continue

            rebuild_failures += 1
            warn("Guided offspring decoder failed to rebuild a valid optimizer; skipping child.")
            self._buffer_decoder_replay_dict(graph_dict)
            if last_valid_graph_dict is not None:
                self._buffer_decoder_replay_dict(last_valid_graph_dict)

        if empty_graph_count:
            warn(
                f"Guided offspring decoder generated {empty_graph_count} empty graphs across {total_requested} decode attempts."
            )
        if duplicate_count:
            warn(
                f"Guided offspring decoder encountered {duplicate_count} duplicate graphs while producing {total_requested} requests."
            )
        if rebuild_failures:
            warn(f"Guided offspring decoder rebuild failures: {rebuild_failures}")
        if missing_slot_rejections:
            warn(f"Guided offspring decoder graphs missing optimizer output coverage: {missing_slot_rejections}")
        if inactive_optimizer_count:
            warn(f"Guided offspring optimizers with no parameter updates: {inactive_optimizer_count}")
        if repair_failures:
            warn(f"Guided offspring repair hook could not rescue {repair_failures} decoded graphs.")

        if stats:
            sample = ", ".join(f"child {idx}: edges={edges}, params={params}" for idx, edges, params in stats[:10])
            self.reporters.info(f"Guided offspring graph stats (first 10): {sample}")

        if total_requested:
            self.reporters.info(
                "Guided offspring summary: %d/%d survived (duplicates=%d, empty=%d, rebuild_failures=%d, inactive=%d)"
                % (
                    len(stats),
                    total_requested,
                    duplicate_count,
                    empty_graph_count,
                    rebuild_failures,
                    inactive_optimizer_count,
                )
            )

        deduped_genomes, late_duplicates = self._dedupe_genomes_by_signature(new_genomes)
        if late_duplicates:
            duplicate_count += late_duplicates
            warn(f"Guided offspring final dedupe removed {late_duplicates} duplicate graphs.")

        self._accumulate_guided_offspring_stats(total_requested, len(deduped_genomes), invalid_reason_counts)
        return deduped_genomes

    def _dedupe_genomes_by_signature(self, genomes: Sequence[OptimizerGenome]) -> Tuple[List[OptimizerGenome], int]:
        seen: Set[str] = set()
        unique: List[OptimizerGenome] = []
        removed = 0
        for genome in genomes:
            graph_dict = getattr(genome, "graph_dict", None)
            signature = graph_signature_from_dict(graph_dict) if graph_dict else None
            if signature and signature in seen:
                removed += 1
                continue
            if signature:
                seen.add(signature)
            unique.append(genome)
        return unique, removed

    def _graph_output_slot_coverage(self, graph_dict):
        output_keys = list(getattr(self.config.genome_config, "output_keys", []))
        if not output_keys:
            return True, {"missing_slots": [], "attributes": {}, "referenced_slots": []}

        edge_index = graph_dict.get("edge_index")
        if edge_index is None:
            details = {"missing_slots": output_keys, "attributes": {}, "referenced_slots": []}
            return False, details

        try:
            if isinstance(edge_index, torch.Tensor):
                edge_tensor = edge_index.detach().long()
            else:
                edge_tensor = torch.as_tensor(edge_index, dtype=torch.long)
        except Exception:
            details = {"missing_slots": output_keys, "attributes": {}, "referenced_slots": []}
            return False, details

        if edge_tensor.dim() == 1:
            edge_tensor = edge_tensor.view(2, -1)
        elif edge_tensor.dim() != 2 or edge_tensor.size(0) != 2:
            details = {"missing_slots": output_keys, "attributes": {}, "referenced_slots": []}
            return False, details

        referenced_nodes = set()
        if edge_tensor.numel() > 0:
            referenced_nodes.update(int(v) for v in edge_tensor[1].flatten().tolist())

        node_attrs = graph_dict.get("node_attributes") or []
        slot_to_outputs: Dict[int, List[int]] = {}
        slot_misfits: Dict[int, List[int]] = {}
        for idx in range(len(node_attrs)):
            attrs = node_attrs[idx] or {}
            slot_val = attrs.get("pin_slot_index")
            try:
                slot = int(slot_val)
            except (TypeError, ValueError):
                continue
            role = self._decode_pin_role_value(attrs.get("pin_role"))
            if role is None and isinstance(attrs.get("node_type"), str):
                role = _normalize_pin_role(attrs.get("node_type"))
            if role == PIN_ROLE_OUTPUT:
                slot_to_outputs.setdefault(slot, []).append(idx)
            else:
                slot_misfits.setdefault(slot, []).append(idx)

        missing_slots: List[int] = []
        wrong_type_slots: List[int] = []
        for raw_slot in output_keys:
            try:
                slot = int(raw_slot)
            except (TypeError, ValueError):
                continue
            nodes = slot_to_outputs.get(slot, [])
            if not nodes:
                if slot in slot_misfits:
                    wrong_type_slots.append(slot)
                else:
                    missing_slots.append(slot)
                continue
            if not any(node in referenced_nodes for node in nodes):
                missing_slots.append(slot)

        combined_missing = sorted(set(missing_slots))
        attr_lookup: Dict[int, List[str]] = {}
        for slot in combined_missing + sorted(set(wrong_type_slots)):
            names: List[str] = []
            for idx in slot_to_outputs.get(slot, []) + slot_misfits.get(slot, []):
                attrs = node_attrs[idx] or {}
                if isinstance(attrs, dict):
                    names.extend(attribute_key_to_name(name) for name in attrs.keys())
            attr_lookup[slot] = sorted(set(names))

        details = {
            "missing_slots": combined_missing,
            "attributes": attr_lookup,
            "referenced_slots": sorted(referenced_nodes),
            "wrong_type_slots": sorted(set(wrong_type_slots)),
        }
        return len(combined_missing) == 0 and not wrong_type_slots, details

    @staticmethod
    def _edge_list_from_index(edge_index, node_count: int) -> List[Tuple[int, int]]:
        if edge_index is None:
            return []
        if isinstance(edge_index, torch.Tensor):
            tensor = edge_index.clone().detach().long()
        else:
            tensor = torch.as_tensor(edge_index, dtype=torch.long)
        if tensor.dim() == 1:
            tensor = tensor.view(2, -1)
        edges: List[Tuple[int, int]] = []
        if tensor.numel() > 0:
            for src, dst in tensor.t().tolist():
                if 0 <= int(src) < node_count and 0 <= int(dst) < node_count:
                    edges.append((int(src), int(dst)))
        return edges

    @staticmethod
    def _normalize_node_attributes(node_attrs: List[dict] | None, node_count: int) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        node_attrs = node_attrs or []
        for idx in range(node_count):
            attrs = node_attrs[idx] if idx < len(node_attrs) else {}
            normalized_dict: Dict[str, Any] = {}
            if isinstance(attrs, dict):
                for key, value in attrs.items():
                    normalized_dict[attribute_key_to_name(key)] = value
            normalized.append(normalized_dict)
        return normalized

    def _decode_pin_role_value(self, value: Any) -> str | None:
        role = _normalize_pin_role(value) if isinstance(value, str) else None
        if role or value is None:
            return role
        if torch.is_tensor(value):
            vec = value.detach().flatten().float()
            if vec.numel() == 0:
                return None
            best_role = None
            best_score = float("-inf")
            vocab = getattr(self, "shared_attr_vocab", None)
            if vocab is None:
                return None
            embeddings = getattr(vocab, "embedding", None)
            if embeddings is None:
                return None
            for candidate in (PIN_ROLE_INPUT, PIN_ROLE_OUTPUT, PIN_ROLE_HIDDEN):
                idx = vocab.name_to_index.get(candidate)
                if idx is None or idx >= embeddings.weight.size(0):
                    continue
                ref = embeddings.weight[idx].detach().flatten().float()
                size = min(vec.numel(), ref.numel())
                if size == 0:
                    continue
                score = F.cosine_similarity(vec[:size].unsqueeze(0), ref[:size].unsqueeze(0)).item()
                if score > best_score:
                    best_score = score
                    best_role = candidate
            return best_role
        return None

    def _repair_graph_dict(self, graph_dict) -> bool:
        graph_dict.pop("_repair_failure_reason", None)
        graph_dict.pop("_repair_failure_details", None)

        def _record_repaired_graph_snapshot() -> None:
            cloned = self._clone_graph_dict(graph_dict, include_history=False)
            if cloned is not None:
                graph_dict[REPAIRED_GRAPH_DICT_KEY] = cloned

        node_count, edges = self._prepare_decoded_graph_dict(graph_dict)
        configured_inputs = list(getattr(self.config.genome_config, "input_keys", []))
        configured_outputs = list(getattr(self.config.genome_config, "output_keys", []))

        def _extend_node_entries(amount: int) -> None:
            if amount <= 0:
                return
            node_attrs = graph_dict.get("node_attributes") or []
            if not isinstance(node_attrs, list):
                node_attrs = list(node_attrs)
            node_attrs.extend({} for _ in range(amount))
            graph_dict["node_attributes"] = node_attrs
            node_types = graph_dict.get("node_types")
            if isinstance(node_types, torch.Tensor):
                padding = torch.zeros(amount, dtype=node_types.dtype)
                graph_dict["node_types"] = torch.cat([node_types, padding], dim=0)
            elif isinstance(node_types, list):
                node_types.extend([0] * amount)
            elif node_types is None:
                graph_dict["node_types"] = torch.zeros(amount, dtype=torch.long)

        required_inputs = len(configured_inputs)
        expected_input_slots = [order for order, _ in enumerate(configured_inputs)]
        expected_output_slots = [int(ok) for ok in configured_outputs if isinstance(ok, (int, float))]
        minimum_nodes = max(node_count, self._minimum_required_node_count())
        if minimum_nodes <= 0:
            minimum_nodes = 1
        if minimum_nodes > node_count:
            _extend_node_entries(minimum_nodes - node_count)
            node_count = minimum_nodes

        if node_count <= 0:
            _record_repaired_graph_snapshot()
            return False
        edge_set = set(edges)
        adjacency_out = {idx: [] for idx in range(node_count)}
        adjacency_in = {idx: [] for idx in range(node_count)}
        for src, dst in edges:
            if 0 <= src < node_count and 0 <= dst < node_count:
                adjacency_out[src].append(dst)
                adjacency_in[dst].append(src)

        normalized_attrs = self._normalize_node_attributes(graph_dict.get("node_attributes"), node_count)

        def _grow_nodes(amount: int) -> None:
            nonlocal node_count
            if amount <= 0:
                return
            start = node_count
            _extend_node_entries(amount)
            node_count += amount
            for idx in range(start, node_count):
                adjacency_out[idx] = []
                adjacency_in[idx] = []
                normalized_attrs.append({})

        def _candidate_order(indices: Sequence[int], *, key=None, reverse: bool = False) -> List[int]:
            seq = list(indices)
            if self.repair_randomize_connections and len(seq) > 1:
                self._repair_rng.shuffle(seq)
                return seq
            if key is not None:
                seq.sort(key=key, reverse=reverse)
            else:
                seq.sort(reverse=reverse)
            return seq

        def _maybe_shuffle(values: Sequence[int]) -> List[int]:
            seq = list(values)
            if self.repair_randomize_connections and len(seq) > 1:
                self._repair_rng.shuffle(seq)
            return seq

        def _flag_enabled(value: Any) -> bool:
            """Interpret decoder-emitted attributes that may be tensors or scalars."""
            if value is None:
                return False
            if isinstance(value, bool):
                return value
            if torch.is_tensor(value):
                if value.numel() == 0:
                    return False
                try:
                    scalar = value.detach().cpu().view(-1)[0].item()
                except Exception:
                    return False
                return bool(scalar)
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"false", "0", "no", "off"}:
                    return False
                if lowered in {"true", "1", "yes", "on"}:
                    return True
            return bool(value)

        def _node_pin_role(idx: int) -> str | None:
            if idx < 0 or idx >= len(normalized_attrs):
                return None
            attrs = normalized_attrs[idx] or {}
            role = self._decode_pin_role_value(attrs.get("pin_role"))
            if role is None:
                node_type = attrs.get("node_type")
                if isinstance(node_type, str):
                    role = _normalize_pin_role(node_type)
            if role is not None:
                return role
            raw_role = attrs.get("pin_role")
            if raw_role is not None and isinstance(raw_role, str):
                warn(f"Unknown node pin role: {raw_role}")
            return None

        def _node_pin_slot(idx: int) -> int | None:
            if idx < 0 or idx >= len(normalized_attrs):
                return None
            attrs = normalized_attrs[idx] or {}
            raw = attrs.get("pin_slot_index")
            if raw is None:
                return None
            if torch.is_tensor(raw):
                if raw.numel() == 0:
                    return None
                try:
                    value = float(raw.detach().cpu().view(-1)[0].item())
                except Exception:
                    return None
            else:
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    return None
            slot = int(round(value))
            if slot < 0:
                return None
            return slot

        def _assign_pin_role(idx: int, role: str) -> bool:
            if idx < 0 or idx >= len(normalized_attrs):
                return False
            attrs = dict(normalized_attrs[idx] or {})
            current = _normalize_pin_role(attrs.get("pin_role"))
            if _flag_enabled(attrs.get("_pin_role_locked")):
                return current == role
            if current == role:
                return False
            attrs["pin_role"] = role
            normalized_attrs[idx] = attrs
            return True

        def _assign_pin_slot(idx: int, slot: int | None) -> bool:
            if idx < 0 or idx >= len(normalized_attrs):
                return False
            attrs = dict(normalized_attrs[idx] or {})
            current = _node_pin_slot(idx)
            if _flag_enabled(attrs.get("_pin_slot_locked")):
                return current == slot
            if slot is None:
                if "pin_slot_index" in attrs:
                    attrs.pop("pin_slot_index", None)
                    normalized_attrs[idx] = attrs
                    return True
                return False
            slot = int(slot)
            if current == slot and "pin_slot_index" in attrs:
                return False
            attrs["pin_slot_index"] = slot
            normalized_attrs[idx] = attrs
            return True

        def _synthesize_pin_roles_from_slots() -> None:
            for idx in range(len(normalized_attrs)):
                slot = _node_pin_slot(idx)
                if slot is None:
                    continue
                role = _node_pin_role(idx)
                if slot in expected_input_slots and role != PIN_ROLE_INPUT:
                    _assign_pin_role(idx, PIN_ROLE_INPUT)
                    _assign_pin_slot(idx, slot)
                elif slot in expected_output_slots and role != PIN_ROLE_OUTPUT:
                    _assign_pin_role(idx, PIN_ROLE_OUTPUT)
                    _assign_pin_slot(idx, slot)

        _synthesize_pin_roles_from_slots()

        def _recompute_pin_nodes() -> Tuple[List[int], List[int]]:
            inputs = [idx for idx in range(len(normalized_attrs)) if _node_pin_role(idx) == PIN_ROLE_INPUT]
            outputs = [idx for idx in range(len(normalized_attrs)) if _node_pin_role(idx) == PIN_ROLE_OUTPUT]
            inputs.sort(key=lambda idx: (_node_pin_slot(idx) is None, _node_pin_slot(idx) or 0, idx))
            outputs.sort(key=lambda idx: (_node_pin_slot(idx) is None, _node_pin_slot(idx) or 0, idx))
            return inputs, outputs

        input_nodes, output_nodes = _recompute_pin_nodes()

        def _output_slot_coverage() -> Dict[int, int]:
            coverage: Dict[int, int] = {}
            for idx in range(len(normalized_attrs)):
                if _node_pin_role(idx) != PIN_ROLE_OUTPUT:
                    continue
                slot = _node_pin_slot(idx)
                if slot is None:
                    continue
                slot = int(slot)
                if slot not in expected_output_slots:
                    continue
                coverage.setdefault(slot, idx)
            return coverage

        def _input_slot_state() -> Tuple[Dict[int, int], List[int], List[int], List[int], List[int]]:
            slot_to_node: Dict[int, int] = {}
            missing_slots: List[int] = []
            wrong_role_slots: List[int] = []
            flexible_inputs: List[int] = []
            duplicate_slots: List[int] = []

            for idx in input_nodes:
                slot = _node_pin_slot(idx)
                attrs = normalized_attrs[idx] or {}
                if slot is None or slot not in expected_input_slots:
                    flexible_inputs.append(idx)
                    continue
                slot = int(slot)
                if slot in slot_to_node:
                    if not _flag_enabled(attrs.get("_pin_slot_locked")):
                        duplicate_slots.append(slot)
                        _assign_pin_slot(idx, None)
                        flexible_inputs.append(idx)
                    continue
                slot_to_node[slot] = idx

            for idx in range(node_count):
                slot = _node_pin_slot(idx)
                if slot is None or slot not in expected_input_slots:
                    continue
                role = _node_pin_role(idx)
                attrs = normalized_attrs[idx] or {}
                if role != PIN_ROLE_INPUT:
                    if role == PIN_ROLE_OUTPUT:
                        continue
                    if _flag_enabled(attrs.get("_pin_slot_locked")) or _flag_enabled(attrs.get("_pin_role_locked")):
                        if int(slot) not in slot_to_node:
                            wrong_role_slots.append(int(slot))
                    else:
                        _assign_pin_slot(idx, None)
                        if _normalize_pin_role(attrs.get("pin_role")) == PIN_ROLE_INPUT and not _flag_enabled(
                            attrs.get("_pin_role_locked")
                        ):
                            attrs.pop("pin_role", None)
                    continue
                slot_to_node[int(slot)] = idx

            missing_slots = [slot for slot in expected_input_slots if slot not in slot_to_node]
            return slot_to_node, missing_slots, wrong_role_slots, flexible_inputs, duplicate_slots

        def _consume_slot(slots: List[int], preferred: int | None = None) -> int | None:
            if preferred is not None and preferred in slots:
                slots.remove(preferred)
                return preferred
            if slots:
                return slots.pop(0)
            return None

        def _apply_input_fallback() -> None:
            nonlocal input_nodes, output_nodes
            if required_inputs <= 0:
                return
            slot_state = _input_slot_state()
            _, missing_slots, wrong_role_slots, flexible_inputs, _ = slot_state
            missing_slots = sorted(missing_slots)

            def _mark_input(idx: int, slot: int | None) -> None:
                _assign_pin_role(idx, PIN_ROLE_INPUT)
                if slot is not None:
                    _assign_pin_slot(idx, slot)

            if not missing_slots:
                return

            # First, convert nodes that already claim the slot but are mis-typed.
            if wrong_role_slots:
                candidates = list(range(node_count))
                for idx in candidates:
                    slot = _node_pin_slot(idx)
                    if slot is None:
                        continue
                    slot = int(slot)
                    if slot not in missing_slots:
                        continue
                    role = _node_pin_role(idx)
                    if role == PIN_ROLE_INPUT:
                        continue
                    chosen = _consume_slot(missing_slots, slot)
                    if chosen is None:
                        break
                    _mark_input(idx, chosen)

            if not missing_slots and required_inputs:
                input_nodes, output_nodes = _recompute_pin_nodes()
                return

            # Next, reuse existing input nodes that lack slot metadata or collided.
            while missing_slots and flexible_inputs:
                slot = _consume_slot(missing_slots)
                if slot is None:
                    break
                idx = flexible_inputs.pop(0)
                _mark_input(idx, slot)

            if not missing_slots:
                input_nodes, output_nodes = _recompute_pin_nodes()
                return

            # Finally, promote hidden nodes to fill the remaining slots.
            candidate_order = _candidate_order(
                range(node_count),
                key=lambda idx: (
                    len(adjacency_in[idx]) > 0,
                    _node_pin_slot(idx) is None,
                    _node_pin_slot(idx) or 0,
                    idx,
                ),
            )
            for idx in candidate_order:
                if not missing_slots:
                    break
                role = _node_pin_role(idx)
                if role == PIN_ROLE_OUTPUT:
                    continue
                slot = _consume_slot(missing_slots)
                if slot is None:
                    break
                _mark_input(idx, slot)

            input_nodes, output_nodes = _recompute_pin_nodes()

        def _apply_output_fallback() -> None:
            nonlocal input_nodes, output_nodes
            if not expected_output_slots:
                return

            def _mark_output(idx: int, slot: int | None) -> None:
                _assign_pin_role(idx, PIN_ROLE_OUTPUT)
                if slot is not None:
                    _assign_pin_slot(idx, slot)

            slot_to_node = _output_slot_coverage()
            missing_slots = sorted(dict.fromkeys(slot for slot in expected_output_slots if slot not in slot_to_node))
            if not missing_slots:
                return

            reserved_inputs = set(input_nodes)

            # Reuse nodes already claiming a slot when they are mutable.
            for idx in range(node_count):
                if not missing_slots:
                    break
                slot = _node_pin_slot(idx)
                if slot is None or slot not in missing_slots:
                    continue
                if idx in reserved_inputs:
                    continue
                attrs = normalized_attrs[idx] or {}
                if _flag_enabled(attrs.get("_pin_role_locked")) or _flag_enabled(attrs.get("_pin_slot_locked")):
                    continue
                _mark_output(idx, slot)
                missing_slots.remove(slot)

            if not missing_slots:
                input_nodes, output_nodes = _recompute_pin_nodes()
                return

            candidate_order = _candidate_order(
                range(node_count),
                key=lambda idx: (
                    len(adjacency_out[idx]) == 0,
                    _node_pin_slot(idx) is None,
                    _node_pin_slot(idx) or 0,
                    -idx,
                ),
            )

            for idx in candidate_order:
                if not missing_slots:
                    break
                if idx in reserved_inputs:
                    continue
                attrs = normalized_attrs[idx] or {}
                if _flag_enabled(attrs.get("_pin_role_locked")):
                    continue
                slot = _consume_slot(missing_slots)
                if slot is None:
                    break
                _mark_output(idx, slot)

            input_nodes, output_nodes = _recompute_pin_nodes()

            if missing_slots:
                additional = len(missing_slots)
                start_idx = node_count
                _grow_nodes(additional)
                new_idx = start_idx
                for slot in missing_slots:
                    _mark_output(new_idx, slot)
                    new_idx += 1
                missing_slots.clear()
                input_nodes, output_nodes = _recompute_pin_nodes()

        if required_inputs:
            _apply_input_fallback()

        if configured_outputs and len(output_nodes) < len(expected_output_slots):
            _apply_output_fallback()

        if required_inputs:
            slot_state = _input_slot_state()
            _, missing_slots, wrong_role_slots, _, duplicate_slots = slot_state
        else:
            missing_slots = []
            wrong_role_slots = []
            duplicate_slots = []

        if required_inputs and (missing_slots or wrong_role_slots):
            graph_dict["node_attributes"] = normalized_attrs
            graph_dict["_repair_failure_reason"] = "missing_input_slots"
            graph_dict["_repair_failure_details"] = {
                "total_slots": required_inputs,
                "missing_count": len(set(missing_slots) | set(wrong_role_slots)),
                "typed_count": len(input_nodes),
                "missing_slots": [int(slot) for slot in missing_slots],
                "wrong_role_slots": [int(slot) for slot in wrong_role_slots],
                "duplicate_slots": sorted(set(int(slot) for slot in duplicate_slots)),
                "slot_to_node": {int(k): int(v) for k, v in slot_state[0].items()},
                "input_nodes": [int(idx) for idx in input_nodes],
            }
            logger.warning(
                "Missing input slots detected; first attrs=%s",
                normalized_attrs[: min(len(normalized_attrs), max(1, required_inputs + 1))],
            )
            logger.warning(graph_dict)
            _record_repaired_graph_snapshot()
            return False

        if configured_outputs:
            coverage = _output_slot_coverage()
            missing_slots: List[int] = []
            wrong_type_slots: List[int] = []
            for slot in expected_output_slots:
                slot_int = int(slot)
                if slot_int in coverage:
                    continue
                claimants = [idx for idx in range(node_count) if _node_pin_slot(idx) == slot_int]
                if not claimants:
                    missing_slots.append(slot_int)
                else:
                    wrong_type_slots.append(slot_int)
            if missing_slots or wrong_type_slots:
                graph_dict["node_attributes"] = normalized_attrs
                graph_dict["_repair_failure_reason"] = "missing_output_slots"
                graph_dict["_repair_failure_details"] = {
                    "total_slots": len(expected_output_slots),
                    "missing_count": len(missing_slots) + len(wrong_type_slots),
                    "missing_slots": missing_slots,
                    "wrong_type_slots": wrong_type_slots,
                }
                _record_repaired_graph_snapshot()
                return False

        if not input_nodes:
            input_nodes = list(range(min(2, node_count))) or [0]
        if not output_nodes:
            coverage = _output_slot_coverage()
            if coverage:
                output_nodes = list(coverage.values())
            else:
                output_nodes = [node_count - 1]

        hidden_nodes = [idx for idx in range(node_count) if idx not in input_nodes and idx not in output_nodes]
        output_set = set(output_nodes)

        def add_edge(src: int, dst: int) -> bool:
            if src == dst or src < 0 or dst < 0 or src >= node_count or dst >= node_count:
                return False
            key = (int(src), int(dst))
            if key in edge_set:
                return False
            edge_set.add(key)
            edges.append(key)
            adjacency_out[src].append(dst)
            adjacency_in[dst].append(src)
            return True

        target_pool = hidden_nodes or [node for node in range(node_count) if node not in input_nodes]
        target_pool = _maybe_shuffle(target_pool)
        if not target_pool:
            target_pool = output_nodes

        for idx, node in enumerate(input_nodes):
            if adjacency_out[node]:
                continue
            cycle = target_pool if target_pool else [node]
            candidate = cycle[idx % len(cycle)]
            if candidate == node and node_count > 1:
                candidate = (node + 1) % node_count
            add_edge(node, candidate)

        for idx, node in enumerate(output_nodes):
            if adjacency_in[node]:
                continue
            sources = _maybe_shuffle(input_nodes + hidden_nodes)
            if not sources:
                sources = [node]
            candidate = sources[idx % len(sources)]
            if candidate == node and node_count > 1:
                candidate = (node - 1) % node_count
            add_edge(candidate, node)

        def input_reaches_output(src: int) -> bool:
            if src not in adjacency_out:
                return False
            visited: Set[int] = {src}
            queue = deque([src])
            while queue:
                current = queue.popleft()
                if current in output_set:
                    return True
                for dst in adjacency_out.get(current, []):
                    if dst not in visited:
                        visited.add(dst)
                        queue.append(dst)
            return False

        for idx, inp in enumerate(input_nodes):
            if not output_nodes:
                break
            if input_reaches_output(inp):
                continue
            candidate_cycle = _maybe_shuffle(output_nodes)
            candidate = candidate_cycle[idx % len(candidate_cycle)]
            if candidate == inp and node_count > 1:
                candidate = (candidate + 1) % node_count
            added = add_edge(inp, candidate)
            if not added and hidden_nodes:
                hub_cycle = _maybe_shuffle(hidden_nodes)
                hub = hub_cycle[idx % len(hub_cycle)]
                add_edge(inp, hub)
                add_edge(hub, candidate)

        def reachable_nodes() -> Set[int]:
            reachable: Set[int] = set(input_nodes)
            queue = deque(input_nodes)
            while queue:
                current = queue.popleft()
                for dst in adjacency_out.get(current, []):
                    if dst not in reachable:
                        reachable.add(dst)
                        queue.append(dst)
            return reachable

        reachable = reachable_nodes()
        for node in output_nodes:
            if node in reachable:
                continue
            sources = [src for src in reachable if src != node]
            if not sources:
                sources = input_nodes
            if not sources:
                break
            ordered_sources = _candidate_order(sources, key=lambda idx: (len(adjacency_out[idx]), idx))
            add_edge(ordered_sources[0], node)
            reachable = reachable_nodes()

        def nodes_reaching_outputs() -> Set[int]:
            reachable_outputs: Set[int] = set(output_nodes)
            queue = deque(output_nodes)
            while queue:
                current = queue.popleft()
                for src in adjacency_in.get(current, []):
                    if src not in reachable_outputs:
                        reachable_outputs.add(src)
                        queue.append(src)
            return reachable_outputs

        if hidden_nodes:
            fallback_sources = input_nodes or list(range(node_count))
            if not fallback_sources:
                fallback_sources = [0]
            for hidden in hidden_nodes:
                if hidden in reachable:
                    continue
                sources = [src for src in reachable if src != hidden]
                if not sources:
                    sources = fallback_sources
                ordered_sources = _candidate_order(sources, key=lambda idx: (len(adjacency_out[idx]), idx))
                for src in ordered_sources:
                    if src == hidden:
                        continue
                    if add_edge(src, hidden):
                        reachable = reachable_nodes()
                        break

            reverse_reachable = nodes_reaching_outputs()
            fallback_targets = output_nodes or list(range(node_count)) or [0]
            for hidden in hidden_nodes:
                if hidden in reverse_reachable:
                    continue
                targets = [dst for dst in reverse_reachable if dst != hidden]
                if not targets:
                    targets = fallback_targets
                ordered_targets = _candidate_order(targets, key=lambda idx: (len(adjacency_in[idx]), idx))
                for dst in ordered_targets:
                    if dst == hidden:
                        continue
                    if add_edge(hidden, dst):
                        reverse_reachable = nodes_reaching_outputs()
                        break

        graph_dict["node_attributes"] = normalized_attrs

        if not edges:
            default_src = input_nodes[0] if input_nodes else 0
            default_dst = output_nodes[0] if output_nodes else (0 if node_count == 1 else 1)
            if default_src == default_dst and node_count > 1:
                default_dst = (default_dst + 1) % node_count
            add_edge(default_src, default_dst)

        if edges:
            tensor = torch.as_tensor(edges, dtype=torch.long).t().contiguous()
        else:
            tensor = torch.empty((2, 0), dtype=torch.long)
        graph_dict["edge_index"] = tensor
        _record_repaired_graph_snapshot()
        return True if len(edges) > 0 else False

    def _optimizer_updates_parameters(self, optimizer, check_steps=2, delta_eps=1e-12):
        """Run a short dry-run to ensure the optimizer changes model weights."""

        try:
            model = ManyLossMinimaModel(self.task.train_data.num_input_features)
        except Exception:
            return True

        self._reset_optimizer_state(optimizer, None)
        self._reset_optimizer_step_counters(optimizer)

        prev_metrics = None
        named_params = list(model.named_parameters())
        baseline = {name: param.detach().clone() for name, param in named_params}
        max_delta_seen = 0.0

        for _ in range(max(1, check_steps)):
            metrics = self.task.evaluate_metrics(model, self.task.train_data)
            if not torch.isfinite(metrics).all():
                break
            prev = torch.zeros_like(metrics) if prev_metrics is None else prev_metrics
            self._ensure_optimizer_state_shapes(optimizer, named_params)
            try:
                updated_state = optimizer(metrics, prev, named_params)
            except Exception:
                updated_state = None
            if updated_state is None:
                break
            state_dict = self._normalize_state_dict(updated_state, named_params)
            total_delta = 0.0
            for name, before in baseline.items():
                after = state_dict.get(name)
                if after is None:
                    continue
                total_delta += torch.norm(after - before, p=1).item()
            if total_delta > max_delta_seen:
                max_delta_seen = total_delta
            model.load_state_dict(state_dict)
            if total_delta > delta_eps:
                self._reset_optimizer_state(optimizer, None)
                self._reset_optimizer_step_counters(optimizer)
                self._last_optimizer_delta = total_delta
                return True
            prev_metrics = metrics.detach()
            named_params = list(model.named_parameters())
            baseline = {name: param.detach().clone() for name, param in named_params}

        self._reset_optimizer_state(optimizer, None)
        self._reset_optimizer_step_counters(optimizer)
        self._last_optimizer_delta = max_delta_seen
        return False

    def run(self, n=None, offspring_per_species=None):
        """
        Runs NEAT with guided offspring replacing standard reproduction.
        - offspring_per_species: if set, exact number of guided children per species
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        self.generation = 0
        while n is None or self.generation < n:
            self.reporters.start_generation(self.generation)
            self._reset_guided_offspring_stats()
            if not self._decoder_replay_seeded:
                self._seed_decoder_replay_from_population()

            # Evaluate real fitness. Using too few update steps makes every optimizer look identical,
            # so clamp to a minimum to expose behavioral differences early in the run.
            eval_steps = self._generation_eval_steps()
            print(f"Evaluating genomes on {self.task.name()} for {eval_steps} steps")
            self.eval_genomes(list(self.population.items()), self.config, steps=eval_steps)

            # Termination check
            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(self._fitness_value(g) for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    best = max(self.population.values(), key=self._fitness_value)
                    self.reporters.found_solution(self.config, self.generation, best)
                    return best

            # Train surrogate on all evaluated genomes
            valid_graphs, valid_fits = [], []
            invalid_graphs, invalid_fits = [], []
            for gid, genome in self.population.items():
                data = self.genome_to_data(genome)
                if getattr(genome, "invalid_graph", False):
                    invalid_graphs.append(data)
                    invalid_fits.append(genome.fitnesses)
                else:
                    valid_graphs.append(data)
                    valid_fits.append(genome.fitnesses)
            if valid_graphs:
                self.trainer.add_data(valid_graphs, valid_fits)
            if invalid_graphs:
                self.trainer.add_data(
                    invalid_graphs,
                    invalid_fits,
                    invalid_flags=[True] * len(invalid_graphs),
                )

            batch = max(1, len(self.trainer.dataset))
            schedule = self._trainer_epoch_schedule()
            if self.generation == 0:
                mode_note = " (test mode)" if self.test_mode else ""
                self.reporters.info(f"Running initial SCAE warmup ({schedule['epochs']} epochs{mode_note})")
            self.trainer.train(
                epochs=schedule["epochs"],
                batch_size=batch,
                generation=self.generation,
                convex_weight=self.convex_surrogate_weight,
                warmup_epochs=schedule["warmup_epochs"],
                loss_threshold=schedule["loss_threshold"],
                baseline_window=schedule["baseline_window"],
            )
            decoder_epochs, decoder_weight = self._decoder_refresh_schedule()
            replay_payload = self._consume_decoder_replay_graphs()
            if decoder_epochs > 0 and (self.trainer.dataset or replay_payload):
                decoder_batch = min(batch, max(1, len(self.trainer.dataset)))
                empty_ratio = self._guided_empty_ratio()
                self.reporters.info(
                    f"Decoder refresh: epochs={decoder_epochs} weight={decoder_weight:.3f} "
                    f"empty_ratio={empty_ratio:.3f} replay_graphs={len(replay_payload)}"
                )
                self.trainer.decoder_teacher_force_pass(
                    epochs=decoder_epochs,
                    batch_size=decoder_batch,
                    teacher_force_weight=decoder_weight,
                    generation=self.generation,
                    verbose=self.decoder_teacher_verbose,
                    extra_graphs=replay_payload,
                )
            valid_size = len(self.trainer.dataset)
            invalid_size = len(self.trainer.invalid_dataset)
            total_size = valid_size + invalid_size
            self.reporters.info(
                f"Trainer dataset sizes (generation {self.generation}): "
                f"valid={valid_size} invalid={invalid_size} total={total_size}"
            )
            self._emit_dataset_stats(valid_size, invalid_size)

            # Build next‐gen population by species
            self.population = self.reproduction.reproduce(
                self.config, self.species, self.config.pop_size, self.generation, self.task
            )

            # Handle possible extinction
            if not self.species.species:
                self.reporters.complete_extinction()
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type,
                        self.config.genome_config,
                        self.config.pop_size,
                    )
                else:
                    raise CompleteExtinctionException()

            # Re‐speciate and finalize generation
            self.species.speciate(self.config, self.population, self.generation)
            self._adjust_compatibility_threshold(len(self.species.species))
            self._emit_guided_offspring_stats()
            self.reporters.end_generation(self.config, self.population, self.species)

            self.generation += 1

        # if no_fitness_termination
        best = max(self.population.values(), key=self._fitness_value)
        self.reporters.found_solution(self.config, self.generation, best)
        return best

    def _generation_eval_steps(self) -> int:
        base_steps = max(2 * self.generation, 25)
        if self.max_evaluation_steps is not None:
            base_steps = min(base_steps, self.max_evaluation_steps)
        return max(1, int(base_steps))

    def eval_genomes(self, genomes, config, steps=10, epsilon=1e-10):
        """
        Evaluate each genome by using its network as a meta–optimizer.
        """
        metric_keys = self._evaluation_metric_keys(self.task)
        raw_metrics: Dict[int, Dict[str, float]] = {}
        invalid_genomes: List[int] = []
        invalid_reason_counts: Counter = Counter()
        genome_map = {gid: g for gid, g in genomes}

        model = ManyLossMinimaModel(self.task.train_data.num_input_features)
        for genome_id, genome in genomes:
            if getattr(genome, "skip_evaluation", False):
                self._assign_penalty(
                    genome_map[genome_id],
                    reason=getattr(genome, "invalid_reason", "empty_graph"),
                    skip_evaluation=True,
                )
                reason = getattr(genome, "invalid_reason", "empty_graph")
                invalid_reason_counts[reason] += 1
                invalid_genomes.append(genome_id)
                continue
            model_copy = type(model)(self.task.train_data.num_input_features)
            model_copy.load_state_dict(model.state_dict())
            print(f"  Evaluating {genome_id} ({genome.optimizer_path})")
            result = self.evaluate_optimizer(genome.optimizer, model_copy, steps=steps)
            if result is None:
                penalty_metrics = self._assign_penalty(genome_map[genome_id])
                reason = getattr(genome_map[genome_id], "invalid_reason", "invalid_graph")
                invalid_reason_counts[reason] += 1
                invalid_genomes.append(genome_id)
                continue
            setattr(genome_map[genome_id], "invalid_graph", False)
            area_under_metrics, validation_metrics, time_cost, mem_cost = result
            validation_metrics_str = "{" + ";".join([f"{m.name}: {v}" for m, v in validation_metrics.items()]) + "}"
            print(
                f"    Area Under Task Metrics: {area_under_metrics}",
                f"    Validation Metrics: {validation_metrics_str}",
                f"    Time Cost: {time_cost}",
                f"    Memory Cost: {mem_cost}",
            )
            validation_metrics[AreaUnderTaskMetrics] = area_under_metrics
            validation_metrics[TimeCost] = time_cost
            validation_metrics[MemoryCost] = mem_cost

            # Guard against accidental over-long/unnamed fitness vectors by
            # stripping any keys that are not part of the expected metric list.
            filtered_metrics: Dict[Metric, float] = {}
            missing_metrics = []
            for metric in metric_keys:
                value = validation_metrics.get(metric)
                if value is None:
                    missing_metrics.append(metric.name if hasattr(metric, "name") else str(metric))
                    filtered_metrics[metric] = self.INVALID_METRIC_VALUE
                else:
                    filtered_metrics[metric] = value

            extra_keys = [key for key in validation_metrics.keys() if key not in filtered_metrics]
            if missing_metrics:
                warn(
                    "Evaluation skipped Metrics (%s); filling with penalty values." % ", ".join(sorted(missing_metrics))
                )
            if extra_keys:
                warn(
                    "Evaluation produced unexpected Metrics for genome %s: %s; dropping extras."
                    % (
                        genome_id,
                        ", ".join(sorted(str(key) for key in extra_keys)),
                    )
                )

            raw_metrics[genome_id] = filtered_metrics

        if not raw_metrics:
            self.reporters.info(
                f"All genomes invalid this generation ({len(invalid_genomes)}/{len(genome_map)}); skipping Pareto ranking"
            )
            return genomes

        # 3. Pareto front ranking (exclude penalized metrics from dominance calc)
        print("  Calculating Pareto Fronts")
        pareto_metrics: Dict[int, Dict[Metric, float]] = {}
        penalized_for_front: List[int] = []
        for gid, metrics in raw_metrics.items():
            if any(value == self.INVALID_METRIC_VALUE for value in metrics.values()):
                penalized_for_front.append(gid)
            else:
                pareto_metrics[gid] = metrics

        fronts: List[List[int]] = []
        if pareto_metrics:
            fronts = nondominated_sort(pareto_metrics)
        if penalized_for_front:
            fronts.append(penalized_for_front)

        # 4. Compute global min/max per metric using only valid Pareto entries
        mins: Dict[Metric, float] = {}
        maxs: Dict[Metric, float] = {}
        for metric in metric_keys:
            values = [metrics[metric] for metrics in pareto_metrics.values() if metric in metrics]
            if values:
                mins[metric] = min(values)
                maxs[metric] = max(values)
            else:
                mins[metric] = 0.0
                maxs[metric] = 0.0

        # 5. Assign fitness = Pareto rank base + composite normalized score
        for front_idx, front in enumerate(fronts, start=1):
            for genome_id in front:
                # Composite min–max normalized score
                scores = []
                metrics = raw_metrics[genome_id]
                for m in metric_keys:
                    lo, hi = mins[m], maxs[m]
                    if hi - lo < epsilon:
                        norm = 0.0
                    else:
                        v = metrics.get(m, self.INVALID_METRIC_VALUE)
                        if v == self.INVALID_METRIC_VALUE:
                            norm = 0.0
                        elif m.objective == "max":
                            norm = (v - lo) / (hi - lo)
                        else:
                            norm = (hi - v) / (hi - lo)
                    scores.append(norm)
                composite = sum(scores) / len(scores) if scores else 0.0
                # Fitness: higher for earlier fronts, break ties by composite
                genome_map[genome_id].fitness = (len(fronts) - front_idx + 1) + composite
                print(f"    {genome_id}: {genome_map[genome_id].fitness}")
                genome_map[genome_id].fitnesses = raw_metrics[genome_id]
        if invalid_genomes:
            self.reporters.info(
                f"Invalid or skipped guided offspring penalized: {len(invalid_genomes)}/{len(genome_map)}"
            )
        if invalid_reason_counts:
            parts = ", ".join(f"{reason}={count}" for reason, count in sorted(invalid_reason_counts.items()))
            self.reporters.info(f"Invalid guided offspring reasons: {parts}")
        return genomes

    def _assign_penalty(
        self,
        genome,
        reason="invalid_graph",
        skip_evaluation=False,
        penalty_details=None,
        graph_dict=None,
        record_decoder_failure: bool = False,
    ):
        if penalty_details is None:
            penalty_details = getattr(genome, "invalid_penalty_details", None)
        else:
            setattr(genome, "invalid_penalty_details", penalty_details)
        penalty_scale = self._penalty_scale(reason, penalty_details)
        penalty_value = float(self.INVALID_METRIC_VALUE * penalty_scale)
        validation_metrics = {m: (1 if m.objective == "min" else -1) * penalty_value for m in self.task.metrics}
        validation_metrics[AreaUnderTaskMetrics] = penalty_value
        validation_metrics[TimeCost] = penalty_value
        validation_metrics[MemoryCost] = penalty_value
        genome.fitnesses = validation_metrics
        genome.fitness = -0.1 * penalty_scale
        setattr(genome, "invalid_graph", True)
        if skip_evaluation:
            genome.skip_evaluation = True
        genome.invalid_reason = reason
        _INVALID_REASON_COUNTER[reason] += 1
        if record_decoder_failure and graph_dict is not None:
            try:
                self._record_decoder_failure(graph_dict, validation_metrics, reason=reason)
            except Exception as exc:
                warn(f"Failed to store decoder failure for {reason}: {exc}")
        return validation_metrics

    def _penalty_scale(self, reason: str | None, penalty_details: dict | None) -> float:
        reason = reason or "invalid_graph"
        scale = self.INVALID_PENALTY_DEFAULT_SCALE
        if reason == "empty_graph":
            scale = self.INVALID_PENALTY_MAX_SCALE
        elif reason in {"missing_output_slots", "missing_input_slots"}:
            total_slots = None
            if penalty_details:
                total_slots = penalty_details.get("total_slots")
            if not total_slots:
                if reason == "missing_output_slots":
                    total_slots = len(getattr(self.config.genome_config, "output_keys", []))
                else:
                    total_slots = len(getattr(self.config.genome_config, "input_keys", []))
            total_slots = max(1, int(total_slots or 0))
            missing = 0
            if penalty_details:
                missing_count = penalty_details.get("missing_count")
                if missing_count is not None:
                    try:
                        missing = max(0, int(missing_count))
                    except (TypeError, ValueError):
                        missing = 0
                else:
                    slots = set()
                    for slot in penalty_details.get("missing_slots") or []:
                        try:
                            slots.add(int(slot))
                        except (TypeError, ValueError):
                            continue
                    for slot in (
                        penalty_details.get("wrong_type_slots") or penalty_details.get("wrong_role_slots") or []
                    ):
                        try:
                            slots.add(int(slot))
                        except (TypeError, ValueError):
                            continue
                    missing = len(slots)
            miss_ratio = min(1.0, max(0.0, missing / total_slots))
            min_scale = self.MISSING_SLOT_PENALTY_MIN_SCALE
            max_scale = self.MISSING_SLOT_PENALTY_MAX_SCALE
            scale = min_scale + (max_scale - min_scale) * miss_ratio
        elif reason == "inactive_optimizer":
            scale = self.INACTIVE_OPTIMIZER_PENALTY_SCALE
        clamped = max(self.INVALID_PENALTY_MIN_SCALE, min(self.INVALID_PENALTY_MAX_SCALE, scale))
        return float(clamped)

    def evaluate_optimizer(self, optimizer, model, steps=10):
        """
        Runs the optimizer over a number of steps.

        Args:
          optimizer: A TorchScript JIT Graph instance that updates parameters.
          model: The model whose performance is measured by the provided task.
          steps: Number of update iterations.
        """
        # Each evaluation should start from a clean optimizer state so the outcome
        # depends only on the current task/model, not on leftovers from previous
        # genomes that may share the same TorchScript module instance.
        self._reset_optimizer_state(optimizer, None)
        self._reset_optimizer_step_counters(optimizer)
        # TODO: clear all levels of RAM caches in between every run to create fair starting point
        # for comparison
        tracemalloc.start()
        start = time.perf_counter()
        prev_metrics_values = None
        area_under_metrics = 0.0

        invalid_graph = False
        for step in range(steps):
            metrics_values = self.task.evaluate_metrics(model, self.task.train_data)
            if not torch.isfinite(metrics_values).all():
                warn("Task metrics produced NaN/Inf; assigning penalty fitness")
                invalid_graph = True
                break
            if not metrics_values.requires_grad:
                raise RuntimeError(
                    "Task metrics tensor must retain gradient information; ensure evaluate_metrics returns differentiable tensors."
                )
            named_params = list(model.named_parameters())
            # When the underlying task changes between generations the model's
            # parameter shapes can shrink or grow. TorchScript optimizers keep
            # persistent state (e.g., Adam moments) keyed by parameter name, so
            # proactively make sure those buffers match the current shapes
            # before executing the optimizer to avoid shape-mismatch errors.
            self._ensure_optimizer_state_shapes(optimizer, named_params)
            try:
                if prev_metrics_values is None:
                    prev = torch.zeros_like(metrics_values)
                else:
                    prev = prev_metrics_values
                updated_state = optimizer(metrics_values, prev, named_params)
            except (RuntimeError, torch.jit.Error) as err:
                if "INVALID_GRAPH_SHAPE" in str(err):
                    warn("Guided offspring produced invalid graph (shape mismatch); assigning penalty fitness")
                    invalid_graph = True
                    break
                if "size of tensor" in str(err) and "must match" in str(err):
                    warn("Optimizer state mismatch detected; resetting state buffers and retrying")
                    self._reset_optimizer_state(optimizer, named_params)
                    self._ensure_optimizer_state_shapes(optimizer, named_params)
                    metrics_values = self.task.evaluate_metrics(model, self.task.train_data)
                    if not torch.isfinite(metrics_values).all():
                        invalid_graph = True
                        break
                    if not metrics_values.requires_grad:
                        raise RuntimeError(
                            "Task metrics tensor must retain gradient information; ensure evaluate_metrics returns differentiable tensors."
                        )
                    if prev_metrics_values is None:
                        prev = torch.zeros_like(metrics_values)
                    else:
                        prev = prev_metrics_values
                    updated_state = optimizer(metrics_values, prev, named_params)
                else:
                    raise
            state_dict = self._normalize_state_dict(updated_state, named_params)
            if not GuidedPopulation._state_dict_is_finite(state_dict):
                warn("Optimizer produced NaN/Inf parameters; assigning penalty fitness")
                invalid_graph = True
                break
            model.load_state_dict(state_dict)
            prev_metrics_values = self.task.evaluate_metrics(model, self.task.train_data).detach()
            prev_metrics_values = torch.nan_to_num(prev_metrics_values)
            area_under_metrics += float(metrics_values.detach().sum())
        if invalid_graph:
            tracemalloc.stop()
            return None
        stop = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        time_cost = stop - start
        validation_metrics = self.task.evaluate_metrics(model, self.task.valid_data).detach()
        validation_metrics = torch.nan_to_num(validation_metrics)
        validation_metrics = validation_metrics.data.numpy()
        validation_metrics = {m: float(validation_metrics[i]) for i, m in enumerate(self.task.metrics)}
        return area_under_metrics, validation_metrics, time_cost, peak_memory

    def _adjust_compatibility_threshold(self, species_count):
        # Dynamically adapt compatibility threshold to encourage multiple species.
        target = max(2, self.config.pop_size // 10)
        threshold = self.config.species_set_config.compatibility_threshold
        if species_count < target:
            threshold = max(0.1, threshold * 0.9)
            self.reporters.info(f"Reduced compatibility threshold to {threshold:.3f} to encourage speciation")
        elif species_count > target * 1.5:
            threshold *= 1.05
            self.reporters.info(
                f"Increased compatibility threshold to {threshold:.3f} to control species proliferation"
            )
        self.config.species_set_config.compatibility_threshold = threshold

    @staticmethod
    def _ensure_optimizer_state_shapes(optimizer, named_params):
        """Reset or resize any persistent optimizer state tensors to match parameter shapes."""
        # TorchScript modules expose state dictionaries as attributes when annotated.
        state_attrs = GuidedPopulation._optimizer_state_attributes(optimizer)
        if not state_attrs:
            return

        for attr in state_attrs:
            try:
                state_dict = dict(getattr(optimizer, attr))
            except Exception:
                continue

            if not state_dict:
                continue

            normalized = GuidedPopulation._normalize_state_dict(state_dict, named_params)
            updated_state: Dict[str, torch.Tensor] = {}
            for name, param in named_params:
                zero = torch.zeros_like(param)
                cur_val = normalized.get(name)
                if isinstance(cur_val, torch.Tensor) and tuple(cur_val.shape) == tuple(param.shape):
                    val = cur_val.detach().clone()
                else:
                    val = zero
                updated_state[name] = val.detach()

            try:
                setattr(optimizer, attr, updated_state)
            except Exception:
                continue

    @staticmethod
    def _normalize_state_dict(state_dict_like, named_params):
        try:
            state_dict = dict(state_dict_like)
        except Exception:
            raise RuntimeError("Optimizer did not return a mapping of parameter tensors")

        if not named_params:
            return {}

        mapped: Dict[str, torch.Tensor] = {}
        used_keys = set()

        for idx, (name, param) in enumerate(named_params):
            candidates = [name, str(name), str(idx), idx]
            value = None
            for cand in candidates:
                if cand in state_dict:
                    value = state_dict[cand]
                    used_keys.add(cand)
                    break

            if isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(param.shape):
                value = value.detach().clone()
            else:
                value = param.detach().clone()

            mapped[name] = value

        # For any state entries that directly match other parameter names not already filled.
        for key, value in state_dict.items():
            if key in used_keys:
                continue
            if not isinstance(key, str) or key in mapped:
                continue
            if isinstance(value, torch.Tensor):
                mapped[key] = value.detach().clone()

        return mapped

    @staticmethod
    def _optimizer_state_attributes(optimizer):
        cache = GuidedPopulation._optimizer_state_attr_cache
        try:
            cached = cache.get(optimizer)
        except TypeError:
            cached = None
        if cached is not None:
            return cached

        attrs: List[str] = []

        candidate_attrs = set()
        candidate_attrs.update(dir(optimizer))
        code = getattr(optimizer, "code", None)
        if isinstance(code, str):
            candidate_attrs.update(re.findall(r"self\.([A-Za-z0-9_]+)", code))

        for attr in candidate_attrs:
            if not isinstance(attr, str) or attr.startswith("_"):
                continue
            try:
                value = getattr(optimizer, attr)
            except Exception:
                continue
            try:
                mapping = dict(value)
            except Exception:
                continue
            valid_mapping = False
            if mapping:
                if all(isinstance(k, str) for k in mapping.keys()) and all(
                    isinstance(v, torch.Tensor) for v in mapping.values() if v is not None
                ):
                    valid_mapping = True
                else:
                    continue
            else:
                # Accept empty dict-like attributes (TorchScript Dict[str, Tensor]) so we can
                # populate them later even before they've stored any state.
                if isinstance(value, dict):
                    valid_mapping = True
            if valid_mapping:
                attrs.append(attr)
        try:
            cache[optimizer] = attrs
        except TypeError:
            # TorchScript objects may not support weak references; skip caching in that case.
            pass
        return attrs

    @staticmethod
    def _reset_optimizer_state(optimizer, named_params):
        state_attrs = GuidedPopulation._optimizer_state_attributes(optimizer)
        if not state_attrs:
            return
        for attr in state_attrs:
            zero_state: Dict[str, torch.Tensor] = {}
            try:
                setattr(optimizer, attr, zero_state)
            except Exception:
                continue

    @staticmethod
    def _reset_optimizer_step_counters(optimizer):
        candidate_attrs = [
            "step",
            "steps",
            "iteration",
            "iterations",
            "t",
        ]
        for attr in candidate_attrs:
            if not hasattr(optimizer, attr):
                continue
            value = getattr(optimizer, attr)
            try:
                if isinstance(value, torch.Tensor):
                    setattr(optimizer, attr, torch.zeros_like(value))
                elif isinstance(value, (int, float)):
                    setattr(optimizer, attr, type(value)(0))
            except Exception:
                continue

    @staticmethod
    def _state_dict_is_finite(state_dict):
        for value in state_dict.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                return False
        return True
