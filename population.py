from neat.population import Population
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import random
from typing import List
from warnings import warn

from genome import OptimizerGenome
from genes import NODE_TYPE_TO_INDEX, NODE_TYPE_OPTIONS
from search_space_compression import *
from tasks import *


class GuidedPopulation(Population):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_latent_dim = 16
        task_latent_dim = 10
        num_node_types = 10
        fitness_dim = 2
        graph_encoder = GraphEncoder(
            num_node_types=len(NODE_TYPE_OPTIONS),
            node_emb_dim=10,
            hidden_dims=[32, 32],
            latent_dim=graph_latent_dim
        ).to(self.device)
        task_encoder = TasksEncoder(
            hidden_dim=16,
            latent_dim=task_latent_dim,
            type_embedding_dim=max(len(TASK_FEATURE_DIMS)//2, 1)
        ).to(self.device)
        decoder = ARGraphDecoder(latent_dim=graph_latent_dim, hidden_dim=128).to(self.device)
        predictor = FitnessPredictor(
            latent_dim=graph_latent_dim+task_latent_dim,
            hidden_dim=64,
            fitness_dim=fitness_dim
        ).to(self.device)

        self.guide = DAGTaskFitnessRegularizedVAE(graph_encoder, task_encoder, decoder, predictor).to(self.device)
        self.optimizer = torch.optim.Adam(self.guide.parameters(), lr=0.001)
        self.trainer = OnlineTrainer(self.guide, self.optimizer, self.device)

    def genome_to_data(self, genome: OptimizerGenome):
        """
        Convert a NEAT OptimizerGenome into:
          1) a PyG Data object with .x_type and .edge_index
          2) genome.task_type  (str)
          3) genome.task_features  (List[float] or Tensor)
        """
        # 1) x_type: a long tensor of shape [N_nodes]
        #    We sort by node ID so positions line up.
        node_ids = sorted(genome.nodes.keys())
        x_type_list = []
        for nid in node_ids:
            node = genome.nodes[nid]
            # map your NodeGene.node_type string → integer
            idx = NODE_TYPE_TO_INDEX.get(node.node_type)
            if idx is None:
                raise KeyError(f"Unknown node_type {node.node_type!r}")
            x_type_list.append(idx)
            node_features = []
            # Decide a consistent ordering of attribute‐names
            attr_names = sorted(node.dynamic_attributes.keys())
            nums = []
            strs = []
            for name in attr_names:
                val = node.dynamic_attributes.get(name)
                if isinstance(val, (int, float)):
                    nums.append(float(val))
                elif isinstance(val, str):
                    strs.append(self.guide.string_embedder.encode(val, convert_to_tensor=True, device=self.device))
                else:
                    warn('missing attribute type in genome-to-data conversion')
            if len(nums) > 0:
                parts = [torch.tensor(nums, dtype=torch.float, device=self.device)] + strs
            else:
                parts = strs
            if len(parts) > 0:
                node_features.append(torch.cat(parts, dim=0))
            else:
                node_features.append(torch.tensor([]))
        x_type = torch.tensor(x_type_list, dtype=torch.long, device=self.device)
        node_features = torch.stack(node_features, dim=0)
        node_types = torch.tensor(x_type_list, dtype=torch.long, device=self.device)

        # 2) edge_index: collect all enabled connections
        edges = []
        for (src, dst), conn in genome.connections.items():
            if conn.enabled:
                # ensure both src/dst are in our node_ids
                if src in node_ids and dst in node_ids:
                    # we need local indices 0…N-1, so remap via node_ids list
                    local_src = node_ids.index(src)
                    local_dst = node_ids.index(dst)
                    edges.append([local_src, local_dst])
        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long, device=self.device).t().contiguous()
        else:
            # no edges
            edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

        return Data(x=node_types, edge_index=edge_index, node_attributes=node_features)

    def generate_guided_offspring(self,
        task_type: str,
        task_features: List[float],
        n_offspring: int  = 10,
        latent_steps: int = 20,
        latent_lr: float  = 1e-2
    ) -> List[OptimizerGenome]:
        """
        For a fixed (task_type, task_features), optimize `z_g` in latent space to maximize
        the surrogate predictor, decode each optimized z_g back into a DAG, then
        convert those DAGs into new NEAT genomes.
        """
        device = self.device

        # 1) Get the task embedding (mu_t, lv_t) and a fixed z_t
        mu_t, lv_t = self.guide.tasks_encoder([task_type], [task_features])
        z_t = self.guide.reparameterize(mu_t, lv_t, self.guide.tasks_latent_mask)
        # expand to match the number of offspring
        z_t = z_t.expand(n_offspring, -1).clone()   # -> (n_offspring, task_latent_dim)

        # 2) Initialize random graph latents and set requires_grad=True
        graph_latent_dim = self.guide.graph_encoder.latent_dim
        z_g = torch.randn((n_offspring, graph_latent_dim), device=device,
                          requires_grad=True)

        # 3) Optimize z_g via Adam ascent to maximize predictor(z_g, z_t)
        opt = torch.optim.Adam([z_g], lr=latent_lr)
        for _ in range(latent_steps):
            pred = self.guide.fitness_predictor(z_g, z_t)    # -> (n_offspring, fitness_dim)
            # scalarize: e.g. maximize sum over all dims & batch‐mean
            loss = - pred.sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # 4) Decode each optimized latent into adjacency logits & node features
        with torch.no_grad():
            adj_logits, feat_logits = self.guide.decode(z_g)   # → (B, max_nodes, max_nodes), (B, max_nodes, node_feat_dim)
            # threshold edges at 0
            edge_masks = (adj_logits > 0).cpu()

        # 5) Convert each decoded DAG to a NEAT genome
        new_genomes = []
        for i in range(n_offspring):
            # extract adjacency & node types
            adj = edge_masks[i].nonzero(as_tuple=False).tolist()  # list of [src, dst]
            node_types = feat_logits[i].argmax(dim=-1).cpu().tolist()  # class per node

            # build a new genome (stub—adapt to your genome constructor)
            g = self.config.genome_type(0, self.config.genome_config)  # fresh id=0
            # add node genes
            for nid, ntype in enumerate(node_types):
                g.nodes[nid] = self.config.genome_config.create_node(ntype)
            # add connection genes
            for src, dst in adj:
                key = (src, dst)
                cg = self.config.genome_config.create_connection(src, dst)
                cg.enabled = True
                g.connections[key] = cg

            new_genomes.append(g)

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

            task_type = random.choice(list(TASK_FEATURE_DIMS.keys()))
            task = TASK_TYPE_TO_CLASS[task_type].random_init()

            # 1) Evaluate real fitness
            eval_genomes(list(self.population.items()), self.config, task, steps=2*generation)

            # 2) Train surrogate one epoch on all evaluated genomes
            graphs, fits = [], []
            for gid, genome in self.population.items():
                g = self.genome_to_data(genome)
                graphs.append(g)
                # shape [1,fitness_dim]
                fits.append(genome.fitnesses)
            task_train_features, task_valid_features = task.get_features()
            distance = torch.linalg.norm(task_train_features - task_valid_features)
            if distance > len(task_train_features)/2:
                warn(f'distance between training ({task_train_features}) and validation ({task_valid_features}) task features = {distance}')

            self.trainer.add_data(graphs, fits, task_type, task_train_features)

            if generation < gen_for_full_train_resize:
                self.trainer.train(epochs=5, batch_size=len(graphs))
            elif generation > gen_for_full_train_resize:
                self.trainer.train(epochs=1, batch_size=len(graphs))
            else:
                self.trainer.train(warmup_epochs=10, batch_size=len(graphs))

            # 3) Build next‐gen population by species
            new_pop = {}
            for species in self.species.species.values():
                members = species.members
                # sort descending by real fitness
                sorted_members = sorted(members, key=lambda g: g.fitness, reverse=True)
                # 3a) keep top N
                kept = sorted_members[:keep_per_species]

                # 3b) generate guided children
                num_to_make = (offspring_per_species
                               if offspring_per_species is not None
                               else len(members) - keep_per_species)
                guided_kids = self.generate_guided_offspring(
                    task_type     = task_type,
                    task_features = task_features,
                    n_offspring   = num_to_make
                )
                # assign predicted fitness
                for kid in guided_kids:
                    # use surrogate to score
                    g, t = self.genome_to_data(kid)
                    mu_g, lv_g = self.guide.graph_encoder(
                        g.x_type.to(self.device),
                        g.edge_index.to(self.device),
                        torch.zeros(g.num_nodes, dtype=torch.long, device=self.device)
                    )
                    mu_t, lv_t = self.guide.tasks_encoder([t], [[*tfeat]])
                    z_g = self.guide.reparameterize(mu_g, lv_g, self.guide.graph_latent_mask)
                    z_t = self.guide.reparameterize(mu_t, lv_t, self.guide.tasks_latent_mask)
                    pred = self.guide.fitness_predictor(z_g, z_t).mean().item()
                    kid.fitness = pred

                # 3c) fill species pool
                for g in kept + guided_kids:
                    new_pop[g.key] = g

            # 4) Replace population & re‐speciate
            self.population = new_pop
            self.species.speciate(self.config, self.population, self.generation)
            self.reporters.end_generation(self.config, self.population, self.species)

            # termination check
            if not self.config.no_fitness_termination:
                fv = self.fitness_criterion(g.fitness for g in self.population.values())
                if fv >= self.config.fitness_threshold:
                    best = max(self.population.values(), key=lambda g: g.fitness)
                    self.reporters.found_solution(self.config, self.generation, best)
                    return best

            # extinction
            if not self.species.species:
                self.reporters.complete_extinction()
                if self.config.reset_on_extinction:
                    self.population = self.reproduction.create_new(
                        self.config.genome_type, self.config.genome_config, self.config.pop_size
                    )
                else:
                    raise CompleteExtinctionException()

            self.generation += 1

        # if no_fitness_termination
        best = max(self.population.values(), key=lambda g: g.fitness)
        self.reporters.found_solution(self.config, self.generation, best)
        return best

    def eval_genomes(genomes, config, task, steps=10, epsilon=1e-10):
        """
        Evaluate each genome by using its network as a meta–optimizer.
        """
        raw_metrics: Dict[int, Dict[str, float]] = {}
        genome_map = {gid: g for gid, g in genomes}

        model = ManyLossMinimaModel(task.observed_dims)
        for genome_id, genome in genomes:
            model_copy = type(model)(task.observed_dims)
            model_copy.load_state_dict(model.state_dict())
            area_under_metrics, validation_metrics, time_cost, mem_cost = evaluate_optimizer(genome.optimizer, model_copy, task, steps)
            raw_metrics[genome_id] = validation_metrics

        # 2. Prepare metric names & objectives
        metric_names = [m.name for m in task.metrics]
        objectives   = {m.name: m.objective for m in task.metrics}

        # 3. Pareto front ranking
        fronts = nondominated_sort(raw_metrics, objectives)
        num_fronts = len(fronts)

        # 4. Compute global min/max per metric
        mins = {m: min(raw_metrics[g][m] for g in raw_metrics) for m in metric_names}
        maxs = {m: max(raw_metrics[g][m] for g in raw_metrics) for m in metric_names}

        # 5. Assign fitness = Pareto rank base + composite normalized score
        for front_idx, front in enumerate(fronts, start=1):
            for genome_id in front:
                vals = raw_metrics[genome_id]
                # Composite min–max normalized score
                scores = []
                for m in metric_names:
                    lo, hi = mins[m], maxs[m]
                    if hi - lo < epsilon:
                        norm = 1.0
                    else:
                        v = vals[m]
                        if objectives[m] == 'max':
                            norm = (v - lo) / (hi - lo)
                        else:
                            norm = (hi - v) / (hi - lo)
                    scores.append(norm)
                composite = sum(scores) / len(scores)
                # Fitness: higher for earlier fronts, break ties by composite
                genome_map[genome_id].fitness = (num_fronts - front_idx + 1) + composite
                genome.fitnesses = vals

    def evaluate_optimizer(optimizer, model, task, steps=10):
        """
        Runs the optimizer over a number of steps.

        Args:
          optimizer: A TorchScript JIT Graph instance that updates parameters.
          model: The model whose performance is measured by the provided task.
          task: The task on which to evaluate the optimizer.
          steps: Number of update iterations.
        """
        tracemalloc.start()
        start = time.perf_counter()
        prev_metrics_values = torch.tensor([0.0] * len(task.metrics))
        area_under_metrics = 0.0
        for step in range(steps):
            metrics_values = task.evaluate_metrics(model, task.train_data).requires_grad_()
            new_params = optimizer(metrics_values, prev_metrics_values, model.named_parameters())
            model.load_state_dict(new_params)
            prev_metrics_values = task.evaluate_metrics(model, task.train_data).requires_grad_()
            area_under_metrics += np.sum(metrics_values.detach().data.numpy())
        stop = time.perf_counter()
        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        time_cost = stop - start
        validation_metrics = task.evaluate_metrics(model, task.valid_data).detach().data.numpy()
        validation_metrics = { name: float(validation_metrics[i]) for i, name in enumerate([m.name for m in task.metrics]) }
        return area_under_metrics, validation_metrics, time_cost, peak_memory
