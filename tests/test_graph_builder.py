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

    assert original == rebuilt
