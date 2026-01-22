import hashlib
from typing import Dict, List, Optional, Tuple

import torch
import torch._C
import torch.jit
import torch.nn as nn

from genes import NODE_TYPE_OPTIONS, NodeGene
from genome import OptimizerGenome


def _encode_string_sequence(values):
    tokens = [str(v) for v in values if v is not None]
    if not tokens:
        return None
    hashed = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        hashed.append(int.from_bytes(digest[:4], byteorder="little") / 0xFFFFFFFF)
    return torch.tensor(hashed, dtype=torch.float32)


class DynamicOptimizerModule(nn.Module):
    """Simple module that applies the decoded DAG to model metrics/parameters."""

    def __init__(self, genome, input_keys, output_keys):
        super().__init__()
        edges: List[Tuple[int, int]] = []
        for (src, dst), conn in genome.connections.items():
            if conn.enabled:
                edges.append((int(src), int(dst)))
        self.edges = torch.jit.Attribute(edges, List[Tuple[int, int]])
        self.input_keys = torch.jit.Attribute(list(input_keys), List[int])
        self.output_keys = torch.jit.Attribute(list(output_keys), List[int])
        self.num_slots = torch.jit.Attribute(max(len(genome.nodes), len(input_keys)), int)

    def forward(
        self,
        loss: torch.Tensor,
        prev_loss: torch.Tensor,
        named_parameters: List[Tuple[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        base_values = torch.jit.annotate(List[torch.Tensor], [])
        base_values.append(loss)
        base_values.append(prev_loss)
        for _, param in named_parameters:
            base_values.append(param)

        num_slots = max(len(base_values), self.num_slots)
        for src, dst in self.edges:
            if src >= num_slots or dst >= num_slots:
                num_slots = max(num_slots, max(src, dst) + 1)

        slots = torch.jit.annotate(List[torch.Tensor], [])
        for idx in range(num_slots):
            if idx < len(base_values):
                tensor = base_values[idx]
            else:
                tensor = torch.zeros_like(loss)
            slots.append(tensor)

        invalid_edges = 0
        for src, dst in self.edges:
            if dst < len(base_values) or src >= len(slots) or dst >= len(slots):
                continue
            src_val = slots[src]
            dst_val = slots[dst]
            if src_val.size() != dst_val.size():
                invalid_edges += 1
                continue
            slots[dst] = dst_val + src_val

        if invalid_edges > 0:
            raise RuntimeError(f"INVALID_GRAPH_SHAPE: tensor shape mismatch (skipped {invalid_edges} edges)")

        outputs = torch.jit.annotate(Dict[str, torch.Tensor], {})
        for ok in self.output_keys:
            idx = int(ok)
            if 0 <= idx < len(slots):
                outputs[str(idx)] = slots[idx]
            else:
                outputs[str(idx)] = torch.zeros_like(loss)
        return outputs


def genome_from_graph_dict(graph_dict, genome_config, key=None) -> OptimizerGenome:
    """Instantiate an OptimizerGenome's nodes/connections from a decoded graph dict."""
    genome = OptimizerGenome(key)
    genome.nodes = {}
    node_types_val = graph_dict.get("node_types")
    if node_types_val is None:
        raise ValueError("graph_dict missing node_types")
    if isinstance(node_types_val, torch.Tensor):
        node_type_indices = node_types_val.clone().detach().view(-1).tolist()
    else:
        node_type_indices = list(node_types_val)
    node_attrs_seq = graph_dict.get("node_attributes", [])
    for nid, type_idx in enumerate(node_type_indices):
        ng = NodeGene(nid, None)
        attr_dict = node_attrs_seq[nid] if nid < len(node_attrs_seq) else {}
        node_type_name = attr_dict.get("node_type")
        if node_type_name is None:
            try:
                node_type_name = NODE_TYPE_OPTIONS[int(type_idx)]
            except (ValueError, TypeError, IndexError):
                node_type_name = "hidden"
        ng.node_type = node_type_name
        dyn_attrs = dict(attr_dict)
        for seq_key in ("__output_types__", "__input_types__", "__input_kinds__", "__getattr_output_types__"):
            val = dyn_attrs.get(seq_key)
            if isinstance(val, (list, tuple)):
                tensor = _encode_string_sequence(val)
                if tensor is not None:
                    dyn_attrs[seq_key] = tensor
                else:
                    dyn_attrs.pop(seq_key, None)
        dyn_attrs.setdefault("__node_kind__", node_type_name)
        ng.dynamic_attributes = dyn_attrs

        scope = dyn_attrs.get("__scope__")
        if scope is not None:
            ng.scope = str(scope)
        genome.nodes[nid] = ng
    genome.next_node_id = len(genome.nodes)

    genome.connections = {}
    edge_index_val = graph_dict.get("edge_index")
    if edge_index_val is not None:
        if isinstance(edge_index_val, torch.Tensor):
            edge_tensor = edge_index_val.clone().detach().long()
        else:
            edge_tensor = torch.as_tensor(edge_index_val, dtype=torch.long)
        if edge_tensor.dim() == 1:
            edge_tensor = edge_tensor.view(2, -1)
        if edge_tensor.numel() > 0:
            for src, dst in edge_tensor.t().tolist():
                cg = genome.create_connection(genome_config, src, dst)
                cg.enabled = True
                genome.connections[(src, dst)] = cg

    return genome


def rebuild_and_script(graph_dict, config, key, genome: Optional[OptimizerGenome] = None) -> DynamicOptimizerModule:
    """
    1) Rebuild the genome nodes+connections (as before)
    2) Create ScriptModule, attach `w_src_dst` Parameters
    3) Generate the IR with `build_forward_graph` and hook it up
    """
    genome = genome or genome_from_graph_dict(graph_dict, config, key)

    # --- build a Python module and script it ---
    module = DynamicOptimizerModule(genome, config.input_keys, config.output_keys)
    return torch.jit.script(module)
