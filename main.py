import copy
import hashlib
import math
import os
import re

import neat
import torch
import torch.nn as nn

from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import *
from population import *
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import *


def create_initial_genome(config, optimizer):
    """
    Creates an initial genome that mirrors the structure of the provided TorchScript optimizer computation graph.
    """
    genome = config.genome_type(0)

    node_mapping = {}  # from TorchScript nodes to genome node keys
    next_node_id = 0
    for node in optimizer.graph.nodes():
        new_node_gene = NodeGene(next_node_id, node)
        new_node_gene.init_attributes(config.genome_config)
        new_node_gene.dynamic_attributes["__node_kind__"] = node.kind()
        scope = node.scopeName()
        if scope:
            gene.dynamic_attributes["__scope__"] = scope
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
            elif (
                ", %loss.1 : Tensor, %prev_loss : Tensor, %named_parameters.1 : (str, Tensor)[] = prim::Param()"
                not in str(producer)
            ):
                print(f"WARNING: missing mapping for input node [{producer}]")

    genome.connections = connections
    genome.optimizer = optimizer
    return genome


def override_initial_population(population, config):
    """
    Overrides the initial genomes in the population with copies of the exact initial genome.
    """
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
    population.species.speciate(config, population.population, population.generation)


if __name__ == "__main__":
    from genome import OptimizerGenome

    config = neat.Config(
        OptimizerGenome, GuidedReproduction, neat.DefaultSpeciesSet, RelativeRankStagnation, "neat-config"
    )
    population = GuidedPopulation(config)

    override_initial_population(population, config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    num_generations = 1000
    winner = population.run(num_generations)
    print("\nBest genome:\n{!s}".format(winner))
