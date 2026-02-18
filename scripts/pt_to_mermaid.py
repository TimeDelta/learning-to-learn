#!/usr/bin/env python3
"""Convert a TorchScript optimizer (.pt) into a Mermaid graph file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from attributes import BoolAttribute, FloatAttribute, IntAttribute, StringAttribute
from genes import ensure_node_type_registered
from graph_ir import export_script_module_to_graph_ir
from population_visualizer import RenderContext, build_mermaid_graph
from torchscript_utils import serialize_script_module


def optimizer_to_graph_dict(opt: torch.jit.ScriptModule) -> dict:
    node_types: list[int] = []
    node_attrs: list[dict] = []
    edges: list[list[int]] = []
    value_to_node: dict[str, int] = {}
    node_map: dict[torch._C.Node, int] = {}

    def _record_node(node: torch._C.Node, scope: str | None = None) -> int:
        node_id = len(node_types)
        node_map[node] = node_id
        node_types.append(ensure_node_type_registered(node.kind()))
        attrs: dict = {}
        if scope:
            attrs["scope"] = scope
        for name in node.attributeNames():
            kind = node.kindOf(name)
            if kind == "i":
                attrs[IntAttribute(name)] = node.i(name)
            elif kind == "f":
                attrs[FloatAttribute(name)] = node.f(name)
            elif kind == "s":
                attrs[StringAttribute(name)] = node.s(name)
            elif kind == "b":
                attrs[BoolAttribute(name)] = bool(node.i(name))
        node_attrs.append(attrs)
        return node_id

    def _process_block(block: torch._C.Block, scope: str | None = None) -> int | None:
        first_child: int | None = None
        for node in block.nodes():
            node_id = _record_node(node, scope)
            if first_child is None:
                first_child = node_id
            for inp in node.inputs():
                src = value_to_node.get(inp.debugName())
                if src is None:
                    src_node = inp.node()
                    src = node_map.get(src_node)
                if src is not None:
                    edges.append([src, node_id])
            for out in node.outputs():
                value_to_node[out.debugName()] = node_id
            for idx, inner in enumerate(node.blocks()):
                child_first = _process_block(inner, scope=f"{node.kind()}[{idx}]")
                if child_first is not None:
                    edges.append([node_id, child_first])
        return first_child

    _process_block(opt.graph.block())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    node_types_tensor = torch.tensor(node_types, dtype=torch.long)
    graph_ir, module_state = export_script_module_to_graph_ir(opt)
    return {
        "node_types": node_types_tensor,
        "edge_index": edge_index,
        "node_attributes": node_attrs,
        "serialized_module": serialize_script_module(opt),
        "graph_ir": graph_ir,
        "module_state": module_state,
        "module_type": opt._c._type().qualified_name() if hasattr(opt._c._type(), "qualified_name") else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the TorchScript .pt file to visualize.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Destination .mmd file (defaults to <input>.mmd).",
    )
    parser.add_argument("--rankdir", choices=["LR", "TB"], default="LR", help="Mermaid rank direction.")
    parser.add_argument(
        "--max-attr-lines",
        type=int,
        default=None,
        help="Maximum node attribute lines (omit to show all).",
    )
    parser.add_argument(
        "--max-attr-chars",
        type=int,
        default=48,
        help="Maximum characters per attribute summary (default: 48).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    optimizer = torch.jit.load(str(args.input))
    graph = optimizer_to_graph_dict(optimizer)
    entry = {"genome_id": args.input.stem, "graph": graph}
    mermaid = build_mermaid_graph(
        entry,
        context=RenderContext(task="seed"),
        max_attr_lines=args.max_attr_lines,
        max_attr_value_chars=max(args.max_attr_chars, 8),
        rankdir=args.rankdir,
    )

    output_path = args.output or args.input.with_suffix(".mmd")
    output_path.write_text(mermaid)
    print(f"Wrote Mermaid graph to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - simple CLI wrapper
    raise SystemExit(main())
