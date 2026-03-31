from genes import ConnectionGene, NodeGene
from genome import OptimizerGenome
from novelty import NoveltyArchive, NoveltyMetric, graph_behavior_descriptor


def _make_genome(node_types):
    genome = OptimizerGenome(0)
    for idx, node_type in enumerate(node_types):
        node = NodeGene(idx)
        node.node_type = node_type
        genome.nodes[idx] = node
    for idx in range(len(node_types) - 1):
        conn = ConnectionGene((idx, idx + 1))
        genome.connections[(idx, idx + 1)] = conn
    return genome


def test_graph_descriptor_changes_with_structure():
    genome = _make_genome(["aten::add", "aten::mul", "aten::add"])
    descriptor_a = graph_behavior_descriptor(genome)
    genome.nodes[1].node_type = "aten::sub"
    genome.connections[(0, 1)].enabled = False
    descriptor_b = graph_behavior_descriptor(genome)
    assert descriptor_a != descriptor_b


def test_novelty_archive_scores_and_updates():
    archive = NoveltyArchive(k=2, max_size=8, min_fill=0, insertion_probability=1.0)
    descriptors = {
        0: (0.0, 0.0, 1.0, 0.0, 1.0),
        1: (0.5, 0.5, 1.0, 0.1, 0.0),
        2: (1.0, 0.0, 0.5, 0.0, 0.0),
    }
    scores = archive.score_population(descriptors)
    assert scores[0] != scores[1]
    archive.update(descriptors, scores, valid_ids=[0, 1, 2])
    assert len(archive) == 3


def test_novelty_metric_metadata():
    assert NoveltyMetric.objective == "max"
    assert NoveltyMetric.guidance_weight == 0.0
