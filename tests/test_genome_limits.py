from factories import make_neat_config

from attributes import FloatAttribute
from genes import NodeGene
from genome import OptimizerGenome


def test_mutate_add_node_respects_max_cap():
    config = make_neat_config()
    genome_config = config.genome_config
    genome_config.max_graph_nodes = 2
    genome = OptimizerGenome(1)
    genome.nodes[0] = genome.create_node(genome_config, 0)
    genome.nodes[1] = genome.create_node(genome_config, 1)

    before = len(genome.nodes)
    genome.mutate_add_node(genome_config)

    assert len(genome.nodes) == before, "mutation should not add nodes beyond cap"


def test_node_gene_attribute_cap():
    config = make_neat_config()
    genome_config = config.genome_config
    genome_config.max_attributes_per_node = 1
    node = NodeGene(0)

    assert node.add_attribute(FloatAttribute("alpha"), genome_config) is True
    assert node.add_attribute(FloatAttribute("beta"), genome_config) is False
    assert len(node.dynamic_attributes) == 1
