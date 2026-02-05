import copy
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch._C
import torch.jit
import torch.nn as nn

from genes import NODE_TYPE_OPTIONS, NodeGene
from genome import OptimizerGenome
from torchscript_utils import load_script_module


def _graph_ir_reshape_to_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() <= 1:
        raise RuntimeError("Tensor must be at least 2-D to reshape as matrix")
    first_dim = tensor.shape[0]
    return tensor.reshape(first_dim, -1)


_GRAPH_IR_HELPERS = {"_reshape_to_matrix": torch.jit.script(_graph_ir_reshape_to_matrix)}


class GraphIRBuildError(RuntimeError):
    """Raised when a graph_ir payload cannot be reconstituted."""


class _GraphIRStub(nn.Module):
    """Minimal module so TorchScript provides a mutable Graph shell."""

    def forward(
        self,
        loss: torch.Tensor,
        prev_loss: torch.Tensor,
        named_parameters: List[Tuple[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:  # pragma: no cover - method body replaced at runtime
        return {}


def _encode_string_sequence(values):
    tokens = [str(v) for v in values if v is not None]
    if not tokens:
        return None
    hashed = []
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        hashed.append(int.from_bytes(digest[:4], byteorder="little") / 0xFFFFFFFF)
    return torch.tensor(hashed, dtype=torch.float32)


def _expected_edges_from_graph_dict(graph_dict) -> List[Tuple[int, int]]:
    edge_index_val = graph_dict.get("edge_index")
    if edge_index_val is None:
        return []
    if isinstance(edge_index_val, torch.Tensor):
        edge_tensor = edge_index_val.clone().detach().cpu()
    else:
        edge_tensor = torch.as_tensor(edge_index_val, dtype=torch.long)
    if edge_tensor.numel() == 0:
        return []
    if edge_tensor.dim() == 1:
        edge_tensor = edge_tensor.view(2, -1)
    return list(map(tuple, edge_tensor.t().tolist()))


def _attach_module_metadata(module: torch.jit.ScriptModule, graph_dict, config) -> None:
    module.edges = _expected_edges_from_graph_dict(graph_dict)
    module.edge_parameter_count = len({tuple(edge) for edge in module.edges})
    module.input_keys = list(config.input_keys)
    module.output_keys = list(config.output_keys)
    node_types_val = graph_dict.get("node_types")
    if node_types_val is None:
        module.node_types = torch.empty(0, dtype=torch.long)
    elif isinstance(node_types_val, torch.Tensor):
        module.node_types = node_types_val.clone().detach().cpu()
    else:
        module.node_types = torch.as_tensor(node_types_val, dtype=torch.long)
    # No-op placeholder: TorchScript modules loaded from disk cannot accept new
    # parameters, so we expose the expected parameter count for callers that
    # previously relied on synthesised nn.Parameter objects.


def _clone_module_state_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.clone().detach().cpu()
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    if isinstance(value, list):
        return [_clone_module_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_module_state_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _clone_module_state_value(v) for k, v in value.items()}
    return copy.deepcopy(value)


class _GraphIRBuilder:
    def __init__(self, graph_ir: Dict[str, Any]):
        self.graph_ir = graph_ir
        self.graph: Optional[torch._C.Graph] = None
        self.value_map: Dict[str, torch._C.Value] = {}
        self.attribute_types: Dict[str, torch._C.Type] = {}
        self.type_overrides: Dict[str, str] = {}
        self._type_cache: Dict[str, torch._C.Type] = {}

    def build(self, graph: torch._C.Graph) -> Dict[str, torch._C.Type]:
        self.graph = graph
        self.value_map = {}
        self.attribute_types = {}
        self.type_overrides = {}
        self._clear_graph()
        self._build_block(graph.block(), self.graph_ir, is_graph=True)
        return self.attribute_types

    def _clear_graph(self) -> None:
        assert self.graph is not None
        # Drop outputs first so downstream nodes lose users before destruction.
        while list(self.graph.outputs()):
            self.graph.eraseOutput(0)
        for node in reversed(list(self.graph.nodes())):
            node.destroy()
        while list(self.graph.inputs()):
            self.graph.eraseInput(0)

    def _build_block(self, block: torch._C.Block, block_ir: Dict[str, Any], *, is_graph: bool) -> None:
        for spec in block_ir.get("inputs", []):
            value = self._add_block_input(block, spec, is_graph=is_graph)
            self.value_map[spec["name"]] = value

        for node_ir in block_ir.get("nodes", []):
            node = self._emit_node(block, node_ir)
            for value, spec in zip(node.outputs(), node_ir.get("outputs", [])):
                self._set_value_debug_name(value, spec["name"])
                parsed_type = self._parse_type(spec["type_repr"])
                value.setType(parsed_type)
                self.value_map[spec["name"]] = value
                if spec["type_repr"].startswith("__torch__."):
                    self.type_overrides[value.debugName()] = spec["type_repr"]

        for spec in block_ir.get("outputs", []):
            output_val = self._get_value(spec["name"])
            if is_graph:
                self.graph.registerOutput(output_val)
            else:
                block.registerOutput(output_val)

    def _add_block_input(
        self,
        block: torch._C.Block,
        spec: Dict[str, Any],
        *,
        is_graph: bool,
    ) -> torch._C.Value:
        if is_graph:
            value = self.graph.addInput()
        else:
            value = block.addInputToBlock()
        self._set_value_debug_name(value, spec["name"])
        value.setType(self._parse_type(spec["type_repr"]))
        return value

    def _emit_node(self, block: torch._C.Block, node_ir: Dict[str, Any]) -> torch._C.Node:
        inputs = [self._get_value(name) for name in node_ir.get("inputs", [])]
        node = self.graph.create(node_ir["kind"], inputs, len(node_ir.get("outputs", [])))
        node.insertBefore(block.returnNode())
        self._assign_attributes(node, node_ir.get("attributes", {}))

        for inner_ir in node_ir.get("blocks", []):
            inner_block = node.addBlock()
            self._build_block(inner_block, inner_ir, is_graph=False)

        if node_ir["kind"] in {"prim::GetAttr", "prim::SetAttr"}:
            attr_name = node_ir.get("attributes", {}).get("name")
            outputs = node_ir.get("outputs", [])
            if attr_name and outputs:
                self.attribute_types[str(attr_name)] = self._parse_type(outputs[0]["type_repr"])

        return node

    def _assign_attributes(self, node: torch._C.Node, attributes: Dict[str, Any]) -> None:
        for name, value in attributes.items():
            if isinstance(value, bool):
                node.i_(name, 1 if value else 0)
            elif isinstance(value, int):
                node.i_(name, int(value))
            elif isinstance(value, float):
                node.f_(name, float(value))
            elif isinstance(value, str):
                node.s_(name, value)
            elif torch.is_tensor(value):
                node.t_(name, value.clone().detach().cpu())
            elif isinstance(value, list):
                self._assign_list_attribute(node, name, value)
            elif value is None:
                continue
            else:
                raise GraphIRBuildError(f"Unsupported attribute type for '{name}': {type(value)!r}")

    def _assign_list_attribute(self, node: torch._C.Node, name: str, value: List[Any]) -> None:
        if not value:
            node.ss_(name, [])
            return
        first = value[0]
        if isinstance(first, bool):
            node.is_(name, [1 if v else 0 for v in value])
        elif isinstance(first, int):
            node.is_(name, [int(v) for v in value])
        elif isinstance(first, float):
            node.fs_(name, [float(v) for v in value])
        elif isinstance(first, str):
            node.ss_(name, [str(v) for v in value])
        elif torch.is_tensor(first):
            node.ts_(name, [v.clone().detach().cpu() for v in value])
        else:
            raise GraphIRBuildError(f"Unsupported list attribute '{name}' element type: {type(first)!r}")

    def _get_value(self, name: str) -> torch._C.Value:
        try:
            return self.value_map[name]
        except KeyError as exc:
            raise GraphIRBuildError(f"Unknown input '{name}' referenced in graph_ir") from exc

    def _set_value_debug_name(self, value: torch._C.Value, name: str) -> None:
        try:
            value.setDebugName(name)
        except RuntimeError:
            value.setDebugName(f"v_{name}")

    def _parse_type(self, type_repr: str) -> torch._C.Type:
        cached = self._type_cache.get(type_repr)
        if cached is not None:
            return cached
        type_repr = type_repr.strip()
        result: torch._C.Type
        if type_repr == "Tensor":
            result = torch._C.TensorType.get()
        elif type_repr == "Tensor?":
            result = torch._C.OptionalType(torch._C.TensorType.get())
        elif type_repr == "int":
            result = torch._C.IntType.get()
        elif type_repr == "float":
            result = torch._C.FloatType.get()
        elif type_repr == "number":
            result = torch._C.NumberType.get()
        elif type_repr == "bool":
            result = torch._C.BoolType.get()
        elif type_repr == "str":
            result = torch._C.StringType.get()
        elif type_repr == "Device":
            result = torch._C.DeviceObjType.get()
        elif type_repr == "NoneType":
            result = torch._C.NoneType.get()
        elif type_repr.startswith("Optional[") and type_repr.endswith("]"):
            inner = type_repr[len("Optional[") : -1]
            result = torch._C.OptionalType(self._parse_type(inner))
        elif type_repr.startswith("List[") and type_repr.endswith("]"):
            inner = type_repr[len("List[") : -1]
            result = torch._C.ListType(self._parse_type(inner))
        elif type_repr.endswith("[]"):
            inner = type_repr[:-2]
            result = torch._C.ListType(self._parse_type(inner))
        elif type_repr.startswith("Tuple[") and type_repr.endswith("]"):
            payload = type_repr[len("Tuple[") : -1]
            items = self._split_generic_items(payload)
            result = torch._C.TupleType(tuple(self._parse_type(item) for item in items))
        elif type_repr.startswith("Dict[") and type_repr.endswith("]"):
            payload = type_repr[len("Dict[") : -1]
            items = self._split_generic_items(payload)
            if len(items) != 2:
                raise GraphIRBuildError(f"Invalid Dict payload: {payload}")
            key_repr, value_repr = items
            result = torch._C.DictType(self._parse_type(key_repr), self._parse_type(value_repr))
        elif type_repr.startswith("__torch__."):
            # Preserve class-qualified names for debugging, but treat them as Any inside the builder.
            result = torch._C.AnyType.get()
        else:
            raise GraphIRBuildError(f"Unsupported type representation: {type_repr}")
        self._type_cache[type_repr] = result
        return result

    def _split_generic_items(self, payload: str) -> List[str]:
        items: List[str] = []
        depth = 0
        current = []
        for char in payload:
            if char == "," and depth == 0:
                items.append("".join(current).strip())
                current = []
                continue
            if char in "[<":
                depth += 1
            elif char in "]>":
                depth -= 1
            current.append(char)
        if current:
            items.append("".join(current).strip())
        return items


def _build_module_from_graph_ir(
    graph_ir: Dict[str, Any],
    module_state: Optional[Dict[str, Any]],
    module_type: Optional[str],
    key,
) -> torch.jit.ScriptModule:
    module = torch.jit.script(_GraphIRStub())
    builder = _GraphIRBuilder(graph_ir)
    builder.build(module.graph)
    module.graph_builder_type_overrides = builder.type_overrides

    if module_state:
        for name, value in module_state.items():
            setattr(module, name, _clone_module_state_value(value))

    module.original_module_type = module_type or f"__torch__.graph_builder.Rebuilt_{key}"
    return module


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
    serialized = graph_dict.get("serialized_module")
    if serialized is not None:
        genome.serialized_module = serialized
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


def rebuild_and_script(
    graph_dict,
    config,
    key,
    genome: Optional[OptimizerGenome] = None,
) -> torch.jit.ScriptModule:
    """
    1) Rebuild the genome nodes+connections (as before)
    2) Create ScriptModule, attach `w_src_dst` Parameters
    3) Generate the IR with `build_forward_graph` and hook it up
    """
    if graph_dict.get("serialized_module") is not None:
        module = load_script_module(graph_dict["serialized_module"])
        _attach_module_metadata(module, graph_dict, config)
        return module

    graph_ir = graph_dict.get("graph_ir")
    if graph_ir is not None:
        module_state = graph_dict.get("module_state")
        module_type = graph_dict.get("module_type")
        module = _build_module_from_graph_ir(graph_ir, module_state, module_type, key)
        _attach_module_metadata(module, graph_dict, config)
        return module

    genome = genome or genome_from_graph_dict(graph_dict, config, key)

    # --- build a Python module and script it ---
    module = DynamicOptimizerModule(genome, config.input_keys, config.output_keys)
    scripted = torch.jit.script(module)
    _attach_module_metadata(scripted, graph_dict, config)
    return scripted
