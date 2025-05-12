from neat.population import Population
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

import random
from typing import List

from genome import OptimizerGenome
from genes import NODE_TYPE_TO_INDEX
from search_space_compression import *
from tasks import TASK_FEATURE_DIMS


class GuidedPopulation(Population):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        graph_latent_dim = 16
        task_latent_dim = 10
        num_node_types = 10
        fitness_dim = 2
        graph_encoder = GraphEncoder(
            num_node_types=10,
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
        x_type = torch.tensor(x_type_list, dtype=torch.long, device=self.device)

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

        return Data(x_type=x_type, edge_index=edge_index)

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

    def run(self, fitness_function, n=None, keep_per_species=2, offspring_per_species=None):
        """
        Runs NEAT with guided offspring replacing standard reproduction.
        - keep_per_species:   number of top genomes in each species to preserve
        - offspring_per_species: if set, exact number of guided children per species
        """
        if self.config.no_fitness_termination and (n is None):
            raise RuntimeError("Cannot have no generational limit with no fitness termination")

        generation = 0
        while n is None or generation < n:
            generation += 1
            self.reporters.start_generation(self.generation)

            # 1) Evaluate real fitness
            fitness_function(list(self.population.items()), self.config)

            # 2) Train surrogate one epoch on all evaluated genomes
            graphs, fits, types, feats = [], [], [], []
            for gid, genome in self.population.items():
                g, ttype, tfeat = self.genome_to_data(genome)
                graphs.append(g)
                types.append(ttype)
                feats.append(tfeat)
                # shape [1,fitness_dim]
                fits.append([genome.fitness]*self.guide.fitness_predictor.fc2.out_features)
            self.trainer.add_data(graphs, fits, types, feats)
            self.trainer.train(epochs=1, batch_size=len(graphs))

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
                    task_type     = species.representative_task_type,
                    task_features = species.representative_task_features,
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
