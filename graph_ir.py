"""Utilities for serializing TorchScript graphs into Python dictionaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Sequence, Set, Tuple

import torch


@dataclass(frozen=True)
class ValueSpec:
    name: str
    type_repr: str


def _serialize_value(value: torch._C.Value) -> ValueSpec:
    return ValueSpec(name=value.debugName(), type_repr=str(value.type()))


def _clone_state_value(value: Any) -> Any:
    """Deep-clone Torch values so graph dicts remain pure-Python structures."""
    if torch.is_tensor(value):
        return value.clone().detach().cpu()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        cloned = [_clone_state_value(v) for v in value]
        return type(value)(cloned)
    if isinstance(value, dict):
        return {k: _clone_state_value(v) for k, v in value.items()}
    # Fallback: attempt shallow copy
    return value


def _serialize_attribute(node: torch._C.Node, attr_name: str) -> Any:
    kind = node.kindOf(attr_name)
    if kind == "i":
        return node.i(attr_name)
    if kind == "f":
        return node.f(attr_name)
    if kind == "b":
        return node.i(attr_name) != 0
    if kind == "s":
        return node.s(attr_name)
    if kind == "t":
        return node.t(attr_name).clone().detach().cpu()
    if kind == "is":
        return list(node.is_(attr_name))
    if kind == "fs":
        return list(node.fs_(attr_name))
    if kind == "ss":
        return list(node.ss(attr_name))
    if kind == "ts":
        return [t.clone().detach().cpu() for t in node.ts(attr_name)]
    # Unknown attribute type; fall back to Python representation if possible.
    ivalue = node[attr_name]
    try:
        return _clone_state_value(ivalue)
    except Exception:
        return str(ivalue)


def _export_block(block: torch._C.Block, attr_names: Set[str]) -> Dict[str, Any]:
    block_dict: Dict[str, Any] = {
        "inputs": [_serialize_value(v).__dict__ for v in block.inputs()],
        "outputs": [_serialize_value(v).__dict__ for v in block.outputs()],
        "nodes": [],
    }
    for node in block.nodes():
        node_dict: Dict[str, Any] = {
            "kind": node.kind(),
            "inputs": [val.debugName() for val in node.inputs()],
            "outputs": [_serialize_value(out).__dict__ for out in node.outputs()],
        }
        scope = node.scopeName()
        if scope:
            node_dict["scope"] = scope
        attribute_payload: Dict[str, Any] = {}
        for attr_name in node.attributeNames():
            attribute_payload[attr_name] = _serialize_attribute(node, attr_name)
            if node.kind() in {"prim::GetAttr", "prim::SetAttr"} and attr_name == "name":
                attr_names.add(str(node.s(attr_name)))
        if attribute_payload:
            node_dict["attributes"] = attribute_payload
        blocks = [_export_block(inner_block, attr_names) for inner_block in node.blocks()]
        if blocks:
            node_dict["blocks"] = blocks
        block_dict["nodes"].append(node_dict)
    return block_dict


def export_script_module_to_graph_ir(module: torch.jit.ScriptModule) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return (graph_ir, module_state) for the given TorchScript module."""
    attr_names: Set[str] = set()
    graph_ir = _export_block(module.graph.block(), attr_names)
    module_state: Dict[str, Any] = {}
    for name in sorted(attr_names):
        if not hasattr(module, name):
            continue
        module_state[name] = _clone_state_value(getattr(module, name))
    return graph_ir, module_state
