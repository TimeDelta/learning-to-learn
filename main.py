import copy
import math
import neat
import re
import torch
import torch.nn as nn

from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import *
from loss import MSELoss

def eval_genomes(genomes, config, loss_fn, steps=10, epsilon=1e-10):
    """
    Evaluate each genome by using its network as a meta–optimizer.

    Parameters:
      - genomes: list of (genome_id, genome) tuples
      - config: NEAT configuration

    The final fitness is defined as:
         1.0 / (final_global_loss + average_query_loss + epsilon)
    """
    losses = {}
    for genome_id, genome in genomes:
        model_copy = copy.deepcopy(model)
        optimizer = CustomFeedForwardNetwork.create(genome, config)
        # TODO dynamically determine num steps based on generation num
        loss_parts = evaluate_optimizer(optimizer, model_copy, loss_fn)
        losses[genome_id] = loss_parts

    num_dims = len(next(iter(losses.values())))
    max_vals = [max(losses[genome_id][i] for genome_id in losses) for i in range(num_dims)]

    normalized_losses = {}
    for genome_id, genome in genomes:
        loss_parts = losses[genome_id]
        # equal weighting to all parts of the loss function for now
        normalized = sum([loss_parts[i] / max_vals[i] / num_dims for i in range(num_dims)])
        normalized_losses[genome_id] = normalized
        genome.fitness = 1.0 / (final_loss + epsilon)

def evaluate_optimizer(optimizer, model, loss_fn, steps=10):
    """
    Runs the optimizer over a number of steps.

    Args:
      optimizer: An instance of SymbolicOptimizer (or subclass) that updates a parameter.
      model: The model whose performance is measured by loss_fn.
      loss_fn: Function that takes the model (or a parameter) and returns a scalar loss.
      steps: Number of update iterations.

    Returns:
      1/∑_steps_(∑_dimensions_(loss))
    """
    loss = loss_fn(model)
    area_under_loss = loss

    for step in range(steps):
        # TODO measure time_taken
        new_params = optimizer(loss_fn(model), loss, model.named_parameters())
        model.load_state_dict(new_params)
        loss = loss_fn(model)
        area_under_loss += loss
    # TODO add loss over validation data
    # TODO add loss part for num nodes and edges in optimizer

    return [area_under_loss, validation_loss, optimizer_time, optimizer_size]

def eval_genomes_wrapper(genomes, config):
    # TODO: load data
    eval_genomes(genomes, config, MSELoss(data), steps=10)

def create_initial_genome(config, optimizer):
    """
    Creates an initial genome that mirrors the structure of the provided TorchScript optimizer computation graph.
    """
    genome = config.genome_type(0)

    graph = optimizer.graph

    node_mapping = {} # from TorchScript nodes to genome node keys
    next_node_id = 0
    for node in graph.nodes():
        new_node_gene = NodeGene(next_node_id, node)
        if new_node_gene:
            genome.nodes[next_node_id] = new_node_gene
            node_mapping[node] = next_node_id
            next_node_id += 1

    connections = {}
    innovation = 0
    for node in graph.nodes():
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
            else:
                print(f'WARNING: missing mapping for input node [{producer}]')

    genome.connections = connections
    return genome

def override_initial_population(population, config):
    """
    Overrides the initial genomes in the population with copies of the exact initial genome.
    """
    new_population = {}
    optimizers = []
    for optimizer in config.genome_config.optimizers.split(','):
        optimizers.append(torch.jit.load(f'computation_graphs/optimizers/{optimizer}.pt'))
    i = 0
    for key in population.population.keys():
        new_genome = create_initial_genome(config, optimizers[i % len(optimizers)])
        new_genome.key = key
        new_population[key] = new_genome
        i += 1
    population.population = new_population

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
