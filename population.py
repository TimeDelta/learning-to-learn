import random
import time
import tracemalloc
from typing import List
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


class GuidedPopulation(Population):
    def __init__(self, config):
        super().__init__(config)
        graph_latent_dim = 16
        task_latent_dim = 10
        num_node_types = 10
        fitness_dim = 4
        self.shared_attr_vocab = SharedAttributeVocab([], 5)
        attr_encoder = NodeAttributeDeepSetEncoder(
            self.shared_attr_vocab, encoder_hdim=10, aggregator_hdim=20, out_dim=50
        )
        graph_encoder = GraphEncoder(
            len(NODE_TYPE_OPTIONS), attr_encoder, latent_dim=graph_latent_dim, hidden_dims=[32, 32]
        )
        task_encoder = TasksEncoder(
            hidden_dim=16, latent_dim=task_latent_dim, type_embedding_dim=max(len(TASK_FEATURE_DIMS) // 2, 1)
        )
        decoder = GraphDecoder(len(NODE_TYPE_OPTIONS), graph_latent_dim, self.shared_attr_vocab)
        predictor = FitnessPredictor(
            latent_dim=graph_latent_dim + task_latent_dim, hidden_dim=64, fitness_dim=fitness_dim
        )

        self.guide = SelfCompressingFitnessRegularizedDAGVAE(graph_encoder, task_encoder, decoder, predictor)
        self.optimizer = torch.optim.Adam(self.guide.parameters(), lr=0.001)
        self.trainer = OnlineTrainer(self.guide, self.optimizer)
        stagnation = config.stagnation_type(config.stagnation_config, self.reporters)
        self.reproduction = GuidedReproduction(config.reproduction_config, self.reporters, stagnation)
        self.reproduction.guide_fn = self.generate_guided_offspring

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
            self.shared_attr_vocab.add_names([a.name for a in node.dynamic_attributes.keys()])
            node_types.append(idx)
            node_attributes.append(node.dynamic_attributes)
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
        genome.graph_dict = {"node_types": node_types, "edge_index": edge_index, "node_attributes": node_attributes}
        return Data(node_types=node_types, edge_index=edge_index, node_attributes=node_attributes)

    def generate_guided_offspring(
        self,
        task_type: str,
        task_features: List[np.ndarray],
        starting_genomes: List[OptimizerGenome],
        config,
        n_offspring: int = 10,
        latent_steps: int = 50,
        latent_lr: float = 1e-2,
    ) -> List[OptimizerGenome]:
        """
        For a fixed (task_type, task_features), optimize `z_g` in latent space to maximize
        the surrogate predictor, decode each optimized z_g back into a DAG, then
        convert those DAGs into new NEAT genomes.
        """
        # 1) Get the task embedding (mu_t, lv_t) and a fixed z_t
        task_features = [np.concatenate(task_features, axis=0)]
        mu_t, lv_t = self.guide.tasks_encoder(
            torch.tensor([TASK_TYPE_TO_INDEX[task_type]], dtype=torch.long), task_features
        )
        z_t = self.guide.reparameterize(mu_t, lv_t, self.guide.tasks_latent_mask).detach()
        # expand to match the number of offspring
        z_t = z_t.expand(n_offspring, -1).clone()

        # 2) Initialize random graph latents and set requires_grad=True
        graph_latent_dim = self.guide.graph_encoder.latent_dim
        # encode the optimizer graphs from top half of starting_genomes then optimize, rest are cross-species
        sorted_genomes = sorted(starting_genomes, key=lambda g: g.fitness, reverse=True)
        num_encode = n_offspring // 2
        top_genomes = sorted_genomes[:num_encode]
        data_list = []
        for g in top_genomes:
            data_list.append(
                Data(
                    node_types=g.graph_dict["node_types"].clone().detach().long(),
                    edge_index=g.graph_dict["edge_index"].clone().detach().long(),
                    node_attributes=g.graph_dict["node_attributes"],
                )
            )
        batch = Batch.from_data_list(data_list)
        mu_g, lv_g = self.guide.graph_encoder(batch.node_types, batch.edge_index, batch.node_attributes, batch.batch)
        z_g_encoded = self.guide.reparameterize(mu_g, lv_g, self.guide.graph_latent_mask)
        z_g_encoded = z_g_encoded.clone().detach().requires_grad_(True)
        num_random = n_offspring - num_encode
        if num_random > 0:
            z_g_random = torch.randn((num_random, graph_latent_dim), requires_grad=True)
            z_g = torch.cat([z_g_encoded, z_g_random], dim=0).clone().detach().requires_grad_(True)
        else:
            z_g = z_g_encoded

        # 3) Optimize z_g via Adam ascent to maximize predictor(z_g, z_t)
        opt = torch.optim.Adam([z_g], lr=latent_lr)
        for _ in range(latent_steps):
            pred = self.guide.fitness_predictor(z_g, z_t)
            loss = -pred.sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 4) Decode
        with torch.no_grad():
            graphs = self.guide.decode(z_g)
        new_genomes = []
        for i, graph_dict in enumerate(graphs):
            optimizer = rebuild_and_script(graph_dict, self.config.genome_config, key=i)
            if optimizer:
                genome = OptimizerGenome(i)
                genome.optimizer = optimizer
                genome.graph_dict = graph_dict
                new_genomes.append(genome)
        return new_genomes

    def run(self, n=None, keep_per_species=2, offspring_per_species=None):
        """
        Runs NEAT with guided offspring replacing standard reproduction.
        - keep_per_species:   number of top genomes in each species to preserve
        - offspring_per_species: if set, exact number of guided children per species
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        generation = 0
        gen_for_full_train_resize = 25
        while n is None or generation < n:
            generation += 1
            self.reporters.start_generation(self.generation)

            task_type = random.choice(list(TASK_TYPE_TO_CLASS.keys()))
            task = TASK_TYPE_TO_CLASS[task_type].random_init()

            # Evaluate real fitness
            print(f"Evaluating genomes on {task.name()}")
            self.eval_genomes(list(self.population.items()), self.config, task, steps=2 * generation)

            # Termination check
            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    best = max(self.population.values(), key=lambda g: g.fitness)
                    self.reporters.found_solution(self.config, self.generation, best)
                    return best

            # Train surrogate on all evaluated genomes
            graphs, fits = [], []
            for gid, genome in self.population.items():
                graphs.append(self.genome_to_data(genome))
                fits.append(genome.fitnesses)
            self.trainer.add_data(graphs, fits, task_type, task.features)

            if generation < gen_for_full_train_resize:
                self.trainer.train(epochs=5, batch_size=len(graphs))
            elif generation > gen_for_full_train_resize:
                self.trainer.train(epochs=1, batch_size=len(graphs))
            else:
                self.trainer.train(warmup_epochs=10, batch_size=len(graphs))

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
        raw_metrics: Dict[int, Dict[str, float]] = {}
        genome_map = {gid: g for gid, g in genomes}

        model = ManyLossMinimaModel(task.train_data.num_input_features)
        for genome_id, genome in genomes:
            model_copy = type(model)(task.train_data.num_input_features)
            model_copy.load_state_dict(model.state_dict())
            print(f"  Evaluating {genome_id} ({genome.optimizer_path})")
            area_under_metrics, validation_metrics, time_cost, mem_cost = self.evaluate_optimizer(
                genome.optimizer, model_copy, task, steps
            )
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

        # 3. Pareto front ranking
        print("  Calculating Pareto Fronts")
        fronts = nondominated_sort(raw_metrics)

        # 4. Compute global min/max per metric
        mins = {m: min(raw_metrics[g][m] for g in raw_metrics) for m in validation_metrics}
        maxs = {m: max(raw_metrics[g][m] for g in raw_metrics) for m in validation_metrics}

        # 5. Assign fitness = Pareto rank base + composite normalized score
        for front_idx, front in enumerate(fronts, start=1):
            for genome_id in front:
                # Composite min–max normalized score
                scores = []
                for m in validation_metrics:
                    lo, hi = mins[m], maxs[m]
                    if hi - lo < epsilon:
                        norm = 1.0
                    else:
                        v = raw_metrics[genome_id][m]
                        if m.objective == "max":
                            norm = (v - lo) / (hi - lo)
                        else:
                            norm = (hi - v) / (hi - lo)
                    scores.append(norm)
                composite = sum(scores) / len(scores)
                # Fitness: higher for earlier fronts, break ties by composite
                genome_map[genome_id].fitness = (len(fronts) - front_idx + 1) + composite
                print(f"    {genome_id}: {genome_map[genome_id].fitness}")
                genome_map[genome_id].fitnesses = raw_metrics[genome_id]
        return genomes

    def evaluate_optimizer(self, optimizer, model, task, steps=10):
        """
        Runs the optimizer over a number of steps.

        Args:
          optimizer: A TorchScript JIT Graph instance that updates parameters.
          model: The model whose performance is measured by the provided task.
          task: The task on which to evaluate the optimizer.
          steps: Number of update iterations.
        """
        # TODO: find way to correct for time improvements that are solely due to RAM cache tiers
        tracemalloc.start()
        start = time.perf_counter()
        prev_metrics_values = torch.tensor([0.0] * len(task.metrics))
        area_under_metrics = 0.0
        for step in range(steps):
            metrics_values = task.evaluate_metrics(model, task.train_data).requires_grad_()
            new_params = optimizer(metrics_values, prev_metrics_values, model.named_parameters())
            model.load_state_dict(new_params)
            prev_metrics_values = task.evaluate_metrics(model, task.train_data).requires_grad_()
            area_under_metrics += float(metrics_values.detach().sum())
        stop = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        time_cost = stop - start
        validation_metrics = task.evaluate_metrics(model, task.valid_data).detach().data.numpy()
        validation_metrics = {m: float(validation_metrics[i]) for i, m in enumerate(task.metrics)}
        return area_under_metrics, validation_metrics, time_cost, peak_memory
