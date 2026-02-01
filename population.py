import atexit
import copy
import random
import re
import signal
import time
import tracemalloc
import weakref
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence
from warnings import warn

import torch
import torch.nn.functional as F
from neat.population import Population
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from genes import NODE_TYPE_OPTIONS, NODE_TYPE_TO_INDEX
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

    def __init__(self, config):
        super().__init__(config)
        _register_invalid_reason_reporter()
        graph_latent_dim = 16
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
            len(NODE_TYPE_OPTIONS), attr_encoder, latent_dim=graph_latent_dim, hidden_dims=[32, 32]
        )
        decoder = GraphDecoder(len(NODE_TYPE_OPTIONS), graph_latent_dim, self.shared_attr_vocab)
        predictor = FitnessPredictor(latent_dim=graph_latent_dim, hidden_dim=64, fitness_dim=len(self.metric_keys))

        self.guide = SelfCompressingFitnessRegularizedDAGVAE(graph_encoder, decoder, predictor)
        self.optimizer = torch.optim.Adam(self.guide.parameters(), lr=0.001)
        self.trainer = OnlineTrainer(self.guide, self.optimizer, metric_keys=self.metric_keys)
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

    def _reset_guided_offspring_stats(self):
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
            self.reporters.info(f"Guided offspring invalid counts: total={summary['invalid_total']} :: {parts}")
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

    @staticmethod
    def _evaluation_metric_keys(task) -> List[Metric]:
        metrics: List[Metric] = list(task.metrics)
        metrics.extend([AreaUnderTaskMetrics, TimeCost, MemoryCost])
        return sort_metrics_by_name(metrics)

    @staticmethod
    def _metric_best_values(metric_keys: List[Metric]) -> List[float]:
        return [metric_best_value(metric) for metric in metric_keys]

    @staticmethod
    def _metric_guidance_weights(metric_keys: List[Metric]) -> List[float]:
        return [float(getattr(metric, "guidance_weight", 1.0)) for metric in metric_keys]

    def genome_to_data(self, genome: OptimizerGenome):
        # always rebuild graph_dict so that new attributes are captured
        # sort by node id so positions line up
        node_ids = sorted(genome.nodes.keys())
        node_types = []
        node_attributes = []
        for nid in node_ids:
            node = genome.nodes[nid]
            idx = NODE_TYPE_TO_INDEX.get(node.node_type)
            if idx is None:
                raise KeyError(f"Unknown node_type {node.node_type!r}")
            attr_names = [attribute_key_to_name(a) for a in node.dynamic_attributes.keys()]
            self.shared_attr_vocab.add_names(attr_names)
            node_types.append(idx)
            node_attributes.append(copy.deepcopy(node.dynamic_attributes))
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
        genome.graph_dict = graph_dict
        return Data(node_types=node_types, edge_index=edge_index, node_attributes=node_attributes)

    @staticmethod
    def _clone_graph_dict(graph_dict: dict | None) -> dict | None:
        if not graph_dict:
            return None
        cloned = {}
        node_types = graph_dict.get("node_types")
        if node_types is not None:
            cloned["node_types"] = node_types.clone().detach().cpu()
        edge_index = graph_dict.get("edge_index")
        if edge_index is not None:
            cloned["edge_index"] = edge_index.clone().detach().cpu()
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
        return cloned

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
            pred, _ = self.guide.fitness_predictor(z_g)
            canonical = canonical_log_distance(pred, best_tensor)
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
        resample_attempts = 0
        inactive_optimizer_count = 0
        missing_slot_rejections = 0
        debug_dir = Path("debug_guided_offspring")
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_save_limit = 5
        debug_saved = 0
        generation_idx = getattr(self, "generation", -1)
        seen_edge_sets = set()
        total_requested = z_g.size(0)
        max_decode_attempts = max(1, max_decode_attempts)
        invalid_reason_counts: Counter = Counter()

        for i in range(total_requested):
            latent = z_g[i].unsqueeze(0).clone()
            attempt = 0
            accepted = False
            best_nonempty_latent = None
            while attempt < max_decode_attempts:
                attempt += 1
                with torch.no_grad():
                    decoded = self.guide.decode(latent)
                if isinstance(decoded, tuple):
                    decoded_graphs, _ = decoded
                else:
                    decoded_graphs = decoded
                graph_dict = decoded_graphs[0]
                genome = genome_from_graph_dict(graph_dict, self.config.genome_config, key=i)
                num_edges = len(genome.connections)
                num_params = len(genome.connections)

                if num_edges > 0:
                    best_nonempty_latent = latent.detach().clone()

                if num_edges == 0 or debug_saved < debug_save_limit:
                    debug_path = debug_dir / f"gen{generation_idx}_child{i}_attempt{attempt}_edges{num_edges}.pt"
                    torch.save(graph_dict, debug_path)
                    debug_saved += 1

                if num_edges == 0:
                    empty_graph_count += 1
                    if attempt < max_decode_attempts:
                        base = best_nonempty_latent if best_nonempty_latent is not None else latent
                        latent = base + torch.randn_like(base) * decode_jitter_std
                        resample_attempts += 1
                        continue
                    warn(
                        "Guided offspring decoder produced an empty graph (no edges); assigning penalty fitness and skipping evaluation."
                    )
                    genome.graph_dict = graph_dict
                    self._assign_penalty(genome, self.task, reason="empty_graph", skip_evaluation=True)
                    invalid_reason_counts["empty_graph"] += 1
                    new_genomes.append(genome)
                    accepted = True
                    break

                edge_key = tuple(sorted(genome.connections.keys()))
                if edge_key in seen_edge_sets:
                    duplicate_count += 1
                    if attempt < max_decode_attempts:
                        base = best_nonempty_latent if best_nonempty_latent is not None else latent
                        latent = base + torch.randn_like(base) * decode_jitter_std
                        resample_attempts += 1
                        continue
                    warn("Guided offspring decoder produced a duplicate graph; skipping to preserve diversity.")
                    break

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
                    if attempt < max_decode_attempts:
                        self.reporters.info(message + " -- resampling latent")
                        base = best_nonempty_latent if best_nonempty_latent is not None else latent
                        latent = base + torch.randn_like(base) * decode_jitter_std
                        resample_attempts += 1
                        continue
                    warn(message + "; assigning penalty fitness.")
                    genome.graph_dict = graph_dict
                    self._assign_penalty(genome, self.task, reason="missing_output_slots", skip_evaluation=True)
                    invalid_reason_counts["missing_output_slots"] += 1
                    new_genomes.append(genome)
                    break

                optimizer = rebuild_and_script(graph_dict, self.config.genome_config, key=i, genome=genome)
                if optimizer:
                    if not self._optimizer_updates_parameters(optimizer, self.task, check_steps=2):
                        inactive_optimizer_count += 1
                        if attempt < max_decode_attempts:
                            base = best_nonempty_latent if best_nonempty_latent is not None else latent
                            latent = base + torch.randn_like(base) * decode_jitter_std
                            resample_attempts += 1
                            continue
                        warn(
                            "Guided offspring optimizer failed to modify model parameters; skipping child after retries."
                        )
                        genome.graph_dict = graph_dict
                        self._assign_penalty(genome, self.task, reason="inactive_optimizer", skip_evaluation=True)
                        invalid_reason_counts["inactive_optimizer"] += 1
                        new_genomes.append(genome)
                        break

                    genome.optimizer = optimizer
                    genome.graph_dict = graph_dict
                    new_genomes.append(genome)
                    stats.append((i, num_edges, num_params))
                    seen_edge_sets.add(edge_key)
                    accepted = True
                    break

                rebuild_failures += 1
                if attempt < max_decode_attempts:
                    base = best_nonempty_latent if best_nonempty_latent is not None else latent
                    latent = base + torch.randn_like(base) * decode_jitter_std
                    resample_attempts += 1
                    continue
                warn("Guided offspring decoder failed to rebuild a valid optimizer; skipping child.")
                break

            # move to next latent whether or not we accepted anything

        if empty_graph_count:
            warn(
                f"Guided offspring decoder generated {empty_graph_count} empty graphs across {total_requested * max_decode_attempts} decode attempts."
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

        if stats:
            sample = ", ".join(f"child {idx}: edges={edges}, params={params}" for idx, edges, params in stats[:10])
            self.reporters.info(f"Guided offspring graph stats (first 10): {sample}")

        if total_requested:
            self.reporters.info(
                "Guided offspring summary: %d/%d survived (duplicates=%d, empty=%d, rebuild_failures=%d, inactive=%d, resamples=%d)"
                % (
                    len(stats),
                    total_requested,
                    duplicate_count,
                    empty_graph_count,
                    rebuild_failures,
                    inactive_optimizer_count,
                    resample_attempts,
                )
            )

        self._accumulate_guided_offspring_stats(total_requested, len(new_genomes), invalid_reason_counts)
        return new_genomes

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

        referenced = set()
        if edge_tensor.numel() > 0:
            referenced.update(int(v) for v in edge_tensor[1].flatten().tolist())

        missing = [int(ok) for ok in output_keys if int(ok) not in referenced]
        attr_lookup = {}
        node_attrs = graph_dict.get("node_attributes") or []
        for slot in missing:
            if 0 <= slot < len(node_attrs):
                attrs = node_attrs[slot] or {}
                if isinstance(attrs, dict):
                    names = sorted(attribute_key_to_name(name) for name in attrs.keys())
                else:
                    names = []
            else:
                names = []
            attr_lookup[slot] = names

        details = {"missing_slots": missing, "attributes": attr_lookup, "referenced_slots": sorted(referenced)}
        return len(missing) == 0, details

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
            model.load_state_dict(state_dict)
            if total_delta > delta_eps:
                self._reset_optimizer_state(optimizer, None)
                self._reset_optimizer_step_counters(optimizer)
                return True
            prev_metrics = metrics.detach()
            named_params = list(model.named_parameters())
            baseline = {name: param.detach().clone() for name, param in named_params}

        self._reset_optimizer_state(optimizer, None)
        self._reset_optimizer_step_counters(optimizer)
        return False

    def run(self, n=None, offspring_per_species=None):
        """
        Runs NEAT with guided offspring replacing standard reproduction.
        - offspring_per_species: if set, exact number of guided children per species
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        self.generation = 0
        gen_for_full_train_resize = 25
        while n is None or self.generation < n:
            self.reporters.start_generation(self.generation)
            self._reset_guided_offspring_stats()

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
            if self.generation == 0:
                self.reporters.info("Running initial SCAE warmup (100 epochs)")
                self.trainer.train(epochs=100, batch_size=batch, generation=self.generation)
            elif self.generation < gen_for_full_train_resize:
                self.trainer.train(epochs=50, batch_size=batch, generation=self.generation)
            else:
                self.trainer.train(warmup_epochs=25, epochs=10, batch_size=batch, generation=self.generation)
            self._emit_dataset_stats(len(self.trainer.dataset), len(self.trainer.invalid_dataset))

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
                    self.task,
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
            result = self.evaluate_optimizer(genome.optimizer, model_copy, steps)
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

    def _assign_penalty(self, genome, reason="invalid_graph", skip_evaluation=False):
        validation_metrics = {
            m: (1 if m.objective == "min" else -1) * self.INVALID_METRIC_VALUE for m in self.task.metrics
        }
        validation_metrics[AreaUnderTaskMetrics] = self.INVALID_METRIC_VALUE
        validation_metrics[TimeCost] = self.INVALID_METRIC_VALUE
        validation_metrics[MemoryCost] = self.INVALID_METRIC_VALUE
        genome.fitnesses = validation_metrics
        genome.fitness = -0.1
        setattr(genome, "invalid_graph", True)
        if skip_evaluation:
            genome.skip_evaluation = True
        genome.invalid_reason = reason
        _INVALID_REASON_COUNTER[reason] += 1
        return validation_metrics

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
