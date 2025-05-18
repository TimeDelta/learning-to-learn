import copy
import math
import neat
import re
import torch
import torch.nn as nn

import os
import random
import time
import tracemalloc

from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import *
from models import *
from pareto import *
from population import *

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
            elif ', %loss.1 : Tensor, %prev_loss : Tensor, %named_parameters.1 : (str, Tensor)[] = prim::Param()' not in str(producer):
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
    for fname in os.listdir('computation_graphs/optimizers/'):
        if fname.endswith('.pt'):
            optimizers.append(torch.jit.load(f'computation_graphs/optimizers/{fname}'))
            optimizer_paths.append(f'computation_graphs/optimizers/{fname}')
    i = 0
    for key in population.population.keys():
        new_genome = create_initial_genome(config, optimizers[i % len(optimizers)])
        new_genome.key = key
        new_genome.optimizer_path = optimizer_paths[i % len(optimizers)]
        new_population[key] = new_genome
        i += 1
    population.population = new_population
    population.shared_attr_vocab.add_names(ATTRIBUTE_NAMES)
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
    population = GuidedPopulation(config)

    override_initial_population(population, config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    num_generations = 1000
    winner = population.run(num_generations)
    print('\nBest genome:\n{!s}'.format(winner))
