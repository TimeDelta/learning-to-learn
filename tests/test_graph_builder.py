import glob
import os
import pathlib
import sys

import neat
import pytest
import torch

# allow imports from repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compare_encoders import optimizer_to_data
from genome import OptimizerGenome
from graph_builder import rebuild_and_script
from reproduction import GuidedReproduction


def make_config():
    config_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], "neat-config")
    return neat.Config(
        OptimizerGenome,
        GuidedReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )


@pytest.mark.parametrize("pt_path", glob.glob(os.path.join("computation_graphs", "optimizers", "*.pt")))
def test_graph_builder_rebuilds_pt(pt_path):
    original = torch.jit.load(pt_path)
    data = optimizer_to_data(original)
    graph_dict = {
        "node_types": data.node_types,
        "edge_index": data.edge_index,
        "node_attributes": data.node_attributes,
    }

    config = make_config()
    rebuilt = rebuild_and_script(graph_dict, config.genome_config, key=0)

    assert isinstance(rebuilt, torch.jit.ScriptModule)

    expected_edges = set(map(tuple, data.edge_index.t().tolist()))
    assert set(rebuilt.edges) == expected_edges

    assert rebuilt.input_keys == config.genome_config.input_keys
    assert rebuilt.output_keys == config.genome_config.output_keys

    assert len(list(rebuilt.parameters())) == len(expected_edges)
    assert len(rebuilt.node_types) == len(data.node_types)

    # Verify that the rebuilt computation graph is identical to the original
    if str(rebuilt.graph) != str(original.graph):
        print("Original graph:\n", original.graph)
        print("Rebuilt graph:\n", rebuilt.graph)
    assert str(rebuilt.graph) == str(original.graph), (
        "\nOriginal graph:\n" + str(original.graph) +
        "\nRebuilt graph:\n" + str(rebuilt.graph)
    )
