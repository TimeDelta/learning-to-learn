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

from compare_encoders import optimizer_to_graph_dict
from genome import OptimizerGenome
from graph_builder import genome_from_graph_dict, rebuild_and_script
from relative_rank_stagnation import RelativeRankStagnation
from reproduction import GuidedReproduction


def make_config():
    config_path = os.path.join(pathlib.Path(__file__).resolve().parents[1], "neat-config")
    return neat.Config(
        OptimizerGenome,
        GuidedReproduction,
        neat.DefaultSpeciesSet,
        RelativeRankStagnation,
        config_path,
    )


def get_node_signature(node, type_overrides=None):
    type_overrides = type_overrides or {}
    """Return a simplified structural signature for a JIT node."""
    # simple signature includes kind (operator name), types of inputs, and output type
    input_kinds = [inp.node().kind() for inp in node.inputs()]

    # Collect attributes for a more robust comparison.  For constant values we
    # canonicalise the attribute by converting to a python value where possible.
    attributes = {}
    for name in node.attributeNames():
        kind = node.kindOf(name)
        if kind == "i":
            attributes[name] = node.i(name)
        elif kind == "f":
            attributes[name] = node.f(name)
        elif kind == "s":
            attributes[name] = node.s(name)
        elif kind == "t":
            val = node.t(name)
            # convert small tensors to list to make comparison deterministic
            attributes[name] = val.tolist() if val.numel() <= 20 else (str(val.dtype), tuple(val.size()))
        elif kind == "is":
            attributes[name] = tuple(node.is_(name))
        elif kind == "fs":
            attributes[name] = tuple(node.fs(name))
        elif kind == "ss":
            attributes[name] = tuple(node.ss(name))
        elif kind == "ts":
            attributes[name] = tuple(node.ts(name))

    # Handle prim::Constant nodes explicitly to capture their value.
    if node.kind() == "prim::Constant":
        try:
            const_val = node.output().toIValue()
            if isinstance(const_val, torch.Tensor):
                const_val = const_val.tolist()
            attributes["const_value"] = const_val
        except Exception:
            if "name" in attributes:
                attributes["const_value"] = None

    output_types = []
    for out in node.outputs():
        override = type_overrides.get(out.debugName())
        if override is not None:
            output_types.append(override)
        else:
            output_types.append(str(out.type()))
    output_types = tuple(output_types)
    return (
        node.kind(),
        tuple(input_kinds),
        output_types,
        tuple(sorted(attributes.items())),
    )


def compare_jit_graphs_structural(original: torch.jit.ScriptModule, rebuilt: torch.jit.ScriptModule) -> bool:
    original_overrides = getattr(original, "graph_builder_type_overrides", {})
    rebuilt_overrides = getattr(rebuilt, "graph_builder_type_overrides", {})
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
        original_signature = get_node_signature(original_node, original_overrides)
        rebuilt_signature = get_node_signature(rebuilt_node, rebuilt_overrides)

        if original_signature != rebuilt_signature:
            print(f"Signatures differ at node {i}:", file=sys.stderr)
            print(f"  original Node Kind: {original_node.kind()}", file=sys.stderr)
            print(f"  rebuilt Node Kind: {rebuilt_node.kind()}", file=sys.stderr)
            print(f"  original signature: {original_signature}", file=sys.stderr)
            print(f"  rebuilt signature: {rebuilt_signature}", file=sys.stderr)
            return False

        # assumes a consistent order of inputs and that corresponding inputs have corresponding nodes
        for input_idx, (original_input_val, rebeuilt_input_val) in enumerate(
            zip(original_node.inputs(), rebuilt_node.inputs())
        ):
            if original_input_val.node().kind() != rebeuilt_input_val.node().kind():
                print(f"Input kind differs for node {i}, input {input_idx}", file=sys.stderr)
                return False
            # For constant values, also compare the actual constant contents.
            if original_input_val.node().kind() == "prim::Constant":
                try:
                    val1 = original_input_val.toIValue()
                    val2 = rebeuilt_input_val.toIValue()
                except Exception:
                    val1 = val2 = object()
                if isinstance(val1, torch.Tensor):
                    val1 = val1.tolist()
                if isinstance(val2, torch.Tensor):
                    val2 = val2.tolist()
                if val1 != val2:
                    print(f"Constant value differs for node {i}, input {input_idx}", file=sys.stderr)
                    return False

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


@pytest.mark.parametrize("strip_serialized", [False, True])
@pytest.mark.parametrize("pt_path", glob.glob(os.path.join("computation_graphs", "optimizers", "*.pt")))
def test_graph_builder_rebuilds_pt(pt_path, strip_serialized):
    original = torch.jit.load(pt_path)
    graph_dict = dict(optimizer_to_graph_dict(original))
    if strip_serialized:
        graph_dict.pop("serialized_module", None)

    config = make_config()
    rebuilt = rebuild_and_script(graph_dict, config.genome_config, key=0)

    assert isinstance(rebuilt, torch.jit.ScriptModule)

    edge_index = graph_dict["edge_index"]
    expected_edges = set()
    if isinstance(edge_index, torch.Tensor) and edge_index.numel() > 0:
        expected_edges = set(map(tuple, edge_index.t().tolist()))
    assert set(rebuilt.edges) == expected_edges

    assert rebuilt.input_keys == config.genome_config.input_keys
    assert rebuilt.output_keys == config.genome_config.output_keys

    assert rebuilt.edge_parameter_count == len(expected_edges)
    assert len(rebuilt.node_types) == len(graph_dict["node_types"])

    # Verify that the rebuilt computation graph is structurally identical to the original
    assert compare_jit_graphs_structural(original, rebuilt)

    if strip_serialized:
        for name, value in (graph_dict.get("module_state") or {}).items():
            assert hasattr(rebuilt, name)
            rebuilt_value = getattr(rebuilt, name)
            if torch.is_tensor(value):
                assert torch.equal(rebuilt_value, value)
            else:
                assert rebuilt_value == value


def test_genome_from_graph_dict_hydrates_structure():
    config = make_config()
    graph_dict = {
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.long),
        "node_attributes": [
            {"node_type": "aten::add", "foo": torch.tensor([1.0])},
            {"node_type": "aten::mul"},
        ],
    }

    genome = genome_from_graph_dict(graph_dict, config.genome_config, key=7)

    assert genome.key == 7
    assert sorted(genome.nodes.keys()) == [0, 1]
    assert genome.nodes[0].node_type == "aten::add"
    assert genome.nodes[1].node_type == "aten::mul"
    assert (0, 1) in genome.connections
    assert genome.connections[(0, 1)].enabled


def test_optimizer_to_graph_dict_includes_graph_ir():
    original = torch.jit.load(os.path.join("computation_graphs", "optimizers", "adam_backprop.pt"))
    graph_dict = optimizer_to_graph_dict(original)

    graph_ir = graph_dict.get("graph_ir")
    module_state = graph_dict.get("module_state")

    assert isinstance(graph_ir, dict)
    assert "inputs" in graph_ir and len(graph_ir["inputs"]) > 0
    assert "nodes" in graph_ir and len(graph_ir["nodes"]) > 0
    assert isinstance(module_state, dict)
    # Expect at least the "step" attribute for Adam
    assert "step" in module_state
