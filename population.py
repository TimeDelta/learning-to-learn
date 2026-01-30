import atexit
import copy
import random
import re
import signal
import time
import tracemalloc
import weakref
from collections import Counter
from pathlib import Path
from typing import Dict, List
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
        task_latent_dim = 10
        num_node_types = 10
        self.shared_attr_vocab = SharedAttributeVocab([], 50)
        # Keep attribute-name embeddings high-dimensional (50d) while compressing the
        # DeepSet attribute summaries down to 20d per node before DAG attention.
        attr_encoder = NodeAttributeDeepSetEncoder(
            self.shared_attr_vocab, encoder_hdim=10, aggregator_hdim=20, out_dim=20
        )
        graph_encoder = GraphEncoder(
            len(NODE_TYPE_OPTIONS), attr_encoder, latent_dim=graph_latent_dim, hidden_dims=[32, 32]
        )
        task_encoder = TasksEncoder(
            hidden_dim=16, latent_dim=task_latent_dim, type_embedding_dim=max(len(TASK_FEATURE_DIMS) // 2, 1)
        )
        decoder = GraphDecoder(len(NODE_TYPE_OPTIONS), graph_latent_dim, self.shared_attr_vocab)
        predictor = TaskConditionedFitnessPredictor(latent_dim=graph_latent_dim + task_latent_dim, hidden_dim=64)

        self.guide = SelfCompressingFitnessRegularizedDAGVAE(graph_encoder, task_encoder, decoder, predictor)
        self.optimizer = torch.optim.Adam(self.guide.parameters(), lr=0.001)
        self.trainer = OnlineTrainer(self.guide, self.optimizer)
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = GuidedReproduction(config.reproduction_config, self.reporters, stagnation)
        self.reproduction.guide_fn = self.generate_guided_offspring
        self.reproduction.optimizer_validator = lambda optimizer, task: self._optimizer_updates_parameters(
            optimizer, task, check_steps=1
        )
        self._initial_compression_done = False
        self._enable_initial_compression = getattr(config, "enable_initial_compression", False)
        self.guided_stats_callback = None
        self._guided_offspring_stats = None
        self.dataset_stats_callback = None

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

    def generate_guided_offspring(
        self,
        task,
        starting_genomes: List[OptimizerGenome],
        config,
        n_offspring: int = 10,
        latent_steps: int = 50,
        latent_lr: float = 1e-2,
        max_decode_attempts: int = 5,
        decode_jitter_std: float = 0.05,
        latent_tether_weight: float = 1e-3,
    ) -> List[OptimizerGenome]:
        """
        For a fixed (task_type, task_features), optimize `z_g` in latent space to maximize
        the surrogate predictor, decode each optimized z_g back into a DAG, then
        convert those DAGs into new NEAT genomes.
        """
        task_type = task.name()
        task_type_id = TASK_TYPE_TO_INDEX[task_type]
        metric_keys = self._evaluation_metric_keys(task)
        metric_best_values = self._metric_best_values(metric_keys)
        metric_guidance_weights = self._metric_guidance_weights(metric_keys)
        metric_dim = len(metric_keys)
        self.trainer.remember_task_signature(
            task_type, task.features, metric_dim, metric_best_values, metric_guidance_weights
        )
        expected_len = TASK_FEATURE_DIMS[task_type]
        flat_features = flatten_task_features(task.features, expected_len)
        mu_t, lv_t = self.guide.tasks_encoder(torch.tensor([task_type_id], dtype=torch.long), [flat_features])
        z_t = self.guide.reparameterize(mu_t, lv_t, self.guide.tasks_latent_mask).detach()
        z_t = z_t.expand(n_offspring, -1).clone()

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

        total_latents = z_g.size(0)
        if total_latents == 0:
            return []

        head_specs = self.guide.available_task_heads()
        if not head_specs:
            head_specs = [(task_type_id, metric_dim)]
        elif task_type_id not in {tid for tid, _ in head_specs}:
            head_specs.append((task_type_id, metric_dim))

        head_latents = []
        for head_task_id, head_dim in head_specs:
            features = self.trainer.get_task_features(head_task_id)
            if features is None:
                if head_task_id == task_type_id:
                    features = flat_features
                else:
                    continue
            mu_t_head, lv_t_head = self.guide.tasks_encoder(torch.tensor([head_task_id], dtype=torch.long), [features])
            z_t_head = self.guide.reparameterize(mu_t_head, lv_t_head, self.guide.tasks_latent_mask).detach()
            best_values = self.trainer.get_metric_best_values(head_task_id)
            if best_values is not None:
                best_list = list(best_values)
            elif head_task_id == task_type_id:
                best_list = list(metric_best_values)
            else:
                best_list = [0.0] * head_dim
            best_tensor = torch.as_tensor(best_list, dtype=z_g.dtype, device=z_g.device)
            if best_tensor.numel() < head_dim:
                best_tensor = F.pad(best_tensor, (0, head_dim - best_tensor.numel()))
            elif best_tensor.numel() > head_dim:
                best_tensor = best_tensor[:head_dim]
            weight_values = self.trainer.get_metric_guidance_weights(head_task_id)
            if weight_values is not None:
                weight_list = list(weight_values)
            elif head_task_id == task_type_id:
                weight_list = list(metric_guidance_weights)
            else:
                weight_list = [1.0] * head_dim
            weight_tensor = torch.as_tensor(weight_list, dtype=z_g.dtype, device=z_g.device)
            if weight_tensor.numel() < head_dim:
                weight_tensor = F.pad(weight_tensor, (0, head_dim - weight_tensor.numel()), value=1.0)
            elif weight_tensor.numel() > head_dim:
                weight_tensor = weight_tensor[:head_dim]
            head_latents.append(
                (head_task_id, head_dim, z_t_head.expand(total_latents, -1).clone(), best_tensor, weight_tensor)
            )

        if not head_latents:
            head_latents.append(
                (
                    task_type_id,
                    metric_dim,
                    z_t,
                    torch.as_tensor(metric_best_values, dtype=z_g.dtype, device=z_g.device),
                    torch.as_tensor(metric_guidance_weights, dtype=z_g.dtype, device=z_g.device),
                )
            )

        opt = torch.optim.Adam([z_g], lr=latent_lr)
        for _ in range(latent_steps):
            loss_terms = []
            for head_task_id, head_dim, z_t_head, best_tensor, weight_tensor in head_latents:
                pred = self.guide.predict_task_head(head_task_id, head_dim, z_g, z_t_head)
                canonical = canonical_log_distance(pred, best_tensor)
                weighted = canonical.pow(2) * weight_tensor
                loss_terms.append(weighted.sum(dim=1).mean())
            loss = torch.stack(loss_terms).sum()
            if latent_tether_weight > 0:
                tether = F.mse_loss(z_g, z_g_initial)
                loss = loss + latent_tether_weight * tether
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
                    self._assign_penalty(genome, task, reason="empty_graph", skip_evaluation=True)
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

                optimizer = rebuild_and_script(graph_dict, self.config.genome_config, key=i, genome=genome)
                if optimizer:
                    if not self._optimizer_updates_parameters(optimizer, task, check_steps=2):
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
                        self._assign_penalty(genome, task, reason="inactive_optimizer", skip_evaluation=True)
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

    def _optimizer_updates_parameters(self, optimizer, task, check_steps=2, delta_eps=1e-12):
        """Run a short dry-run to ensure the optimizer changes model weights."""

        try:
            model = ManyLossMinimaModel(task.train_data.num_input_features)
        except Exception:
            return True

        self._reset_optimizer_state(optimizer, None)
        self._reset_optimizer_step_counters(optimizer)

        prev_metrics = None
        named_params = list(model.named_parameters())
        baseline = {name: param.detach().clone() for name, param in named_params}

        for _ in range(max(1, check_steps)):
            metrics = task.evaluate_metrics(model, task.train_data)
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

            task_type = random.choice(list(TASK_TYPE_TO_CLASS.keys()))
            task = TASK_TYPE_TO_CLASS[task_type].random_init()

            # Evaluate real fitness. Using too few update steps makes every optimizer look identical,
            # so clamp to a minimum to expose behavioral differences early in the run.
            eval_steps = max(2 * self.generation, 25)
            print(f"Evaluating genomes on {task.name()} for {eval_steps} steps")
            self.eval_genomes(list(self.population.items()), self.config, task, steps=eval_steps)

            # Termination check
            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    best = max(self.population.values(), key=lambda g: g.fitness)
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
                self.trainer.add_data(valid_graphs, valid_fits, task_type, task.features)
            if invalid_graphs:
                self.trainer.add_data(
                    invalid_graphs,
                    invalid_fits,
                    task_type,
                    task.features,
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
                self.config, self.species, self.config.pop_size, self.generation, task
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
        best = max(self.population.values(), key=lambda g: g.fitness)
        self.reporters.found_solution(self.config, self.generation, best)
        return best

    def eval_genomes(self, genomes, config, task, steps=10, epsilon=1e-10):
        """
        Evaluate each genome by using its network as a meta–optimizer.
        """
        metric_keys = self._evaluation_metric_keys(task)
        raw_metrics: Dict[int, Dict[str, float]] = {}
        invalid_genomes: List[int] = []
        invalid_reason_counts: Counter = Counter()
        genome_map = {gid: g for gid, g in genomes}

        model = ManyLossMinimaModel(task.train_data.num_input_features)
        for genome_id, genome in genomes:
            if getattr(genome, "skip_evaluation", False):
                self._assign_penalty(
                    genome_map[genome_id],
                    task,
                    reason=getattr(genome, "invalid_reason", "empty_graph"),
                    skip_evaluation=True,
                )
                reason = getattr(genome, "invalid_reason", "empty_graph")
                invalid_reason_counts[reason] += 1
                invalid_genomes.append(genome_id)
                continue
            model_copy = type(model)(task.train_data.num_input_features)
            model_copy.load_state_dict(model.state_dict())
            print(f"  Evaluating {genome_id} ({genome.optimizer_path})")
            result = self.evaluate_optimizer(genome.optimizer, model_copy, task, steps)
            if result is None:
                penalty_metrics = self._assign_penalty(genome_map[genome_id], task)
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
            raw_metrics[genome_id] = validation_metrics

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

    def _assign_penalty(self, genome, task, reason="invalid_graph", skip_evaluation=False):
        validation_metrics = {m: (1 if m.objective == "min" else -1) * self.INVALID_METRIC_VALUE for m in task.metrics}
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

    def evaluate_optimizer(self, optimizer, model, task, steps=10):
        """
        Runs the optimizer over a number of steps.

        Args:
          optimizer: A TorchScript JIT Graph instance that updates parameters.
          model: The model whose performance is measured by the provided task.
          task: The task on which to evaluate the optimizer.
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
            metrics_values = task.evaluate_metrics(model, task.train_data)
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
                    metrics_values = task.evaluate_metrics(model, task.train_data)
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
            prev_metrics_values = task.evaluate_metrics(model, task.train_data).detach()
            prev_metrics_values = torch.nan_to_num(prev_metrics_values)
            area_under_metrics += float(metrics_values.detach().sum())
        if invalid_graph:
            tracemalloc.stop()
            return None
        stop = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        time_cost = stop - start
        validation_metrics = task.evaluate_metrics(model, task.valid_data).detach()
        validation_metrics = torch.nan_to_num(validation_metrics)
        validation_metrics = validation_metrics.data.numpy()
        validation_metrics = {m: float(validation_metrics[i]) for i, m in enumerate(task.metrics)}
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
