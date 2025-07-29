import glob
import os
import pathlib
import sys

import neat
import pytest
import torch
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

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


def get_node_signature(node):
    # simple signature includes kind (operator name), types of inputs, and output type
    # TODO: for robust comparison, also need to compare attributes and potentially canonicalize constant values
    input_kinds = [inp.node().kind() for inp in node.inputs()]

    attributes = {}
    if node.kind() == "prim::Constant":
        if node.hasAttribute("value"):
            attributes["value"] = node.t("value")
        elif node.hasAttribute("i"):
            attributes["value"] = node.i("i")
        elif node.hasAttribute("f"):
            attributes["value"] = node.f("f")
    # TODO: finish

    return (node.kind(), tuple(input_kinds), node.output().type(), tuple(sorted(attributes.items())))


def compare_jit_graphs_structural(original: torch.jit.ScriptModule, rebuilt: torch.jit.ScriptModule) -> bool:
    original_inputs = list(original.graph.inputs())
    rebuilt_inputs = list(rebuilt.graph.inputs())
    original_outputs = list(original.graph.outputs())
    rebuilt_outputs = list(rebuilt.graph.outputs())
    if len(original_inputs) != len(rebuilt_inputs) or len(original_outputs) != len(rebuilt_outputs):
        print(
            f"Input/output counts differ: original.graph inputs={len(original_inputs)}, outputs={len(original_outputs)} vs rebuilt inputs={len(rebuilt_inputs)}, outputs={len(rebuilt_outputs)}",
            file=sys.stderr,
        )
        return False

    # default iterator for graph.nodes() is typically a topological sort
    original_nodes = list(original.graph.nodes())
    rebuilt_nodes = list(rebuilt.graph.nodes())

    if len(original_nodes) != len(rebuilt_nodes):
        print(
            f"Number of nodes differ: original.graph has {len(original_nodes)} nodes, rebuilt has {len(rebuilt_nodes)} nodes",
            file=sys.stderr,
        )
        return False

    # create mapping from nodes to canonical representation based on signature + inputs
    original_node_map = {}
    rebuilt_node_map = {}
    for i, (original_node, rebuilt_node) in enumerate(zip(original_nodes, rebuilt_nodes)):
        signature1 = get_node_signature(original_node)
        signature2 = get_node_signature(rebuilt_node)

        if signature1 != signature2:
            print(f"Signatures differ at node {i}:", file=sys.stderr)
            print(f"  original.graph Node Kind: {original_node.kind()}", file=sys.stderr)
            print(f"  rebuilt Node Kind: {rebuilt_node.kind()}", file=sys.stderr)
            # TODO: add more detailed diffing here
            return False

        # assumes a consistent order of inputs and that corresponding inputs have corresponding nodes
        for input_idx, (original_input_val, rebeuilt_input_val) in enumerate(
            zip(original_node.inputs(), rebuilt_node.inputs())
        ):
            if original_input_val.node().kind() != rebeuilt_input_val.node().kind():
                print(f"Input kind differs for node {i}, input {input_idx}", file=sys.stderr)
                return False
            # TODO: need to further compare value properties if they are constants or recursively
            # check if the input nodes themselves are structurally equivalent up to that point

    original_params = dict(original.named_parameters())
    rebuilt_params = dict(rebuilt.named_parameters())
    if len(original_params) != len(rebuilt_params):
        print("Parameter counts differ", file=sys.stderr)
        return False
    for name, original_param in original_params.items():
        if name not in rebuilt_params:
            print(f"Parameter '{name}' missing in rebuilt graph", file=sys.stderr)
            return False
        rebuilt_param = rebuilt_params[name]
        if not torch.equal(original_param, rebuilt_param):
            print(f"Parameter '{name}' values differ", file=sys.stderr)
            return False

    if not compare_custom_data(original, rebuilt):
        print("Custom data attributes differ", file=sys.stderr)
        return False

    return True


def compare_custom_data(original: torch.jit.ScriptModule, rebuilt: torch.jit.ScriptModule) -> bool:
    if hasattr(original, "node_types") and hasattr(rebuilt, "node_types"):
        if original.node_types != rebuilt.node_types:
            print("node_types differ", file=sys.stderr)
            return False
    if hasattr(original, "edge_index") and hasattr(rebuilt, "edge_index"):
        if not torch.equal(original.edge_index, rebuilt.edge_index):
            print("edge_index differ", file=sys.stderr)
            return False
    return True


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

    # Verify that the rebuilt computation graph is structurally identical to the original
    assert compare_jit_graphs_structural(rebuilt, original)
