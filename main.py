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


def _encode_string_sequence(values):
    tokens = [str(v) for v in values if v is not None]
    if not tokens:
        return None
    hashed = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        hashed.append(int.from_bytes(digest[:4], byteorder="little") / 0xFFFFFFFF)
    return torch.tensor(hashed, dtype=torch.float32)


def _annotate_node_for_speciation(gene: NodeGene, node: torch._C.Node) -> None:
    """Attach deterministic metadata so speciation can distinguish identical graphs."""
    if node is None:
        return
    attrs = gene.dynamic_attributes
    attrs["__node_kind__"] = node.kind()
    scope = node.scopeName()
    if scope:
        attrs["__scope__"] = scope

    outputs = list(node.outputs())
    attrs["__num_outputs__"] = len(outputs)
    output_tensor = _encode_string_sequence(str(out.type()) for out in outputs)
    if output_tensor is not None:
        attrs["__output_types__"] = output_tensor

    inputs = list(node.inputs())
    attrs["__num_inputs__"] = len(inputs)
    if inputs:
        kind_tensor = _encode_string_sequence(inp.node().kind() for inp in inputs)
        if kind_tensor is not None:
            attrs["__input_kinds__"] = kind_tensor
        type_tensor = _encode_string_sequence(str(inp.type()) for inp in inputs)
        if type_tensor is not None:
            attrs["__input_types__"] = type_tensor
        getattr_types = []
        for inp in inputs:
            src_node = inp.node()
            if src_node.kind() == "prim::GetAttr" and src_node.outputsSize() > 0:
                try:
                    getattr_types.append(str(src_node.output().type()))
                except RuntimeError:
                    continue
        getattr_tensor = _encode_string_sequence(getattr_types)
        if getattr_tensor is not None:
            attrs["__getattr_output_types__"] = getattr_tensor


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
        _annotate_node_for_speciation(new_node_gene, node)
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
