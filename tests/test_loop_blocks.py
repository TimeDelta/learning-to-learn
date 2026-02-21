import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph_ir import export_script_module_to_graph_ir
from loop_blocks import decode_block_payload, register_graph_blocks, snapshot_registry
from main import create_initial_genome
from population import BLOCK_PAYLOAD_ATTR_PREFIX, GuidedPopulation
from search_space_compression import attribute_key_to_name
from tests.test_graph_builder import make_config  # reuse helper


class LoopModule(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = torch.zeros_like(x)
        i = 0
        while int(i) < 3:
            total = total + x * (i + 1)
            i = i + 1
        return total


def test_loop_blocks_register_and_snapshot():
    module = torch.jit.script(LoopModule())
    graph_ir, _ = export_script_module_to_graph_ir(module)
    node_block_map = register_graph_blocks(graph_ir)
    assert node_block_map, "Expected at least one node to register loop blocks"
    block_ids = {bid for bids in node_block_map.values() for bid in bids}
    snapshot = snapshot_registry(block_ids)
    assert snapshot, "Snapshot should capture registered block payloads"
    for block_id in block_ids:
        assert block_id in snapshot


def test_initial_genome_captures_loop_attributes():
    config = make_config()
    module = torch.jit.script(LoopModule())
    genome = create_initial_genome(config, module)
    assert genome.graph_dict.get("block_registry"), "Graph dict should preserve block registry"
    has_block_attr = False
    for node in genome.nodes.values():
        for key in node.dynamic_attributes.keys():
            if attribute_key_to_name(key).startswith("__block_ref_"):
                has_block_attr = True
                break
        if has_block_attr:
            break
    assert has_block_attr, "Node attributes should include block reference markers"


def test_block_payload_attribute_round_trip():
    config = make_config()
    module = torch.jit.script(LoopModule())
    genome = create_initial_genome(config, module)
    payload_tensors = []
    for attrs in genome.graph_dict.get("node_attributes", []):
        for key, value in attrs.items():
            if isinstance(key, str) and key.startswith(BLOCK_PAYLOAD_ATTR_PREFIX):
                payload_tensors.append(value)
    assert payload_tensors, "Loop nodes should expose encoded block payloads"
    decoded = decode_block_payload(payload_tensors[0])
    assert decoded.get("nodes"), "Decoded payload should contain loop body nodes"


def test_guided_population_preserves_block_registry():
    config = make_config()
    module = torch.jit.script(LoopModule())
    genome = create_initial_genome(config, module)
    pop = GuidedPopulation(config)
    pop.genome_to_data(genome)
    graph_dict = genome.graph_dict
    assert graph_dict.get("block_registry"), "genome graph_dict should copy block registry"
    for attrs in graph_dict.get("node_attributes", []):
        if not isinstance(attrs, dict):
            continue
        if any(name.startswith("__block_ref_") for name in map(attribute_key_to_name, attrs.keys())):
            break
    else:
        raise AssertionError("Decoded graph missing block reference attributes")
