import copy
import math
import neat
import re
import torch
import torch.nn as nn

import random
import time
import tracemalloc

from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import *
from models import *
from pareto import *
from tasks import *

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

def eval_genomes_wrapper(genomes, config):
    true_dims=random.randint(1, 10)
    observed_dims=random.randint(1,10)
    metrics=[MSELoss()]
    task = RegressionTask(true_dims, observed_dims, metrics, num_samples=1000, train_ratio=2.0/3.0)
    eval_genomes(genomes, config, task, steps=10)

def create_initial_genome(config, optimizer):
    """
    Creates an initial genome that mirrors the structure of the provided TorchScript optimizer computation graph.
    """
    genome = config.genome_type(0)

    node_mapping = {} # from TorchScript nodes to genome node keys
    next_node_id = 0
    for node in optimizer.graph.nodes():
        new_node_gene = NodeGene(next_node_id, node)
        if new_node_gene:
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
            elif '%self.1 : __torch__.BackpropGD, %loss.1 : Tensor, %prev_loss : Tensor, %named_parameters.1 : (str, Tensor)[] = prim::Param()' not in str(producer):
                print(f'WARNING: missing mapping for input node [{producer}]')

    genome.connections = connections
    genome.optimizer = optimizer
    return genome

def override_initial_population(population, config):
    """
    Overrides the initial genomes in the population with copies of the exact initial genome.
    """
    new_population = {}
    optimizers = []
    optimizer_paths = []
    for optimizer in config.genome_config.optimizers.split(','):
        optimizers.append(torch.jit.load(f'computation_graphs/optimizers/{optimizer}.pt'))
        optimizer_paths.append(f'computation_graphs/optimizers/{optimizer}.pt')
    i = 0
    for key in population.population.keys():
        new_genome = create_initial_genome(config, optimizers[i % len(optimizers)])
        new_genome.key = key
        new_genome.optimizer_path = optimizer_paths[i % len(optimizers)]
        new_population[key] = new_genome
        i += 1
    population.population = new_population
    population.species.speciate(config, population.population, population.generation)

if __name__ == "__main__":
    from genome import OptimizerGenome
    config = neat.Config(
        OptimizerGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        'neat-config'
    )
    population = neat.Population(config)

    override_initial_population(population, config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    num_generations = 1000
    winner = population.run(eval_genomes_wrapper, num_generations)
    print('\nBest genome:\n{!s}'.format(winner))
