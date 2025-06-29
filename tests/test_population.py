import os
import sys
import neat
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from population import GuidedPopulation
from genome import OptimizerGenome
from genes import NODE_TYPE_TO_INDEX, ConnectionGene, NodeGene
from attributes import IntAttribute, FloatAttribute
from tasks import RegressionTask
from reproduction import GuidedReproduction


def make_config():
    config_path = os.path.join(os.path.dirname(__file__), os.pardir, "neat-config")
    return neat.Config(
        OptimizerGenome,
        GuidedReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


def create_simple_genome(key=0):
    genome = OptimizerGenome(key)
    ng0 = NodeGene(0, None)
    ng0.node_type = "aten::add"
    ng0.dynamic_attributes = {IntAttribute("a"): 1}
    ng1 = NodeGene(1, None)
    ng1.node_type = "aten::mul"
    ng1.dynamic_attributes = {FloatAttribute("b"): 0.5}
    genome.nodes = {0: ng0, 1: ng1}
    cg = ConnectionGene((0, 1))
    cg.enabled = True
    genome.connections = {(0, 1): cg}
    genome.next_node_id = 2
    return genome


def test_genome_to_data():
    config = make_config()
    pop = GuidedPopulation(config)
    genome = create_simple_genome()
    data = pop.genome_to_data(genome)

    assert genome.graph_dict is not None
    assert list(data.node_types.tolist()) == [NODE_TYPE_TO_INDEX["aten::add"], NODE_TYPE_TO_INDEX["aten::mul"]]
    assert data.edge_index.size(1) == 1
    assert data.edge_index[:, 0].tolist() == [0, 1]
    assert len(data.node_attributes) == 2
    assert "a" in pop.shared_attr_vocab.name_to_index
    assert "b" in pop.shared_attr_vocab.name_to_index


def test_generate_guided_offspring():
    config = make_config()
    pop = GuidedPopulation(config)
    pop.guide.decoder.max_nodes = 2
    pop.guide.decoder.max_attributes_per_node = 2

    g1 = create_simple_genome(0)
    g1.fitness = 1.0
    g2 = create_simple_genome(1)
    g2.fitness = 0.5
    pop.genome_to_data(g1)
    pop.genome_to_data(g2)

    task = RegressionTask.random_init(num_samples=4, silent=True)
    offspring = pop.generate_guided_offspring(
        task.name(), task.features, [g1, g2], config, n_offspring=2, latent_steps=1
    )

    assert isinstance(offspring, list)
    assert len(offspring) <= 2
    for child in offspring:
        assert isinstance(child, OptimizerGenome)
        assert child.graph_dict is not None
