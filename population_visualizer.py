"""Utilities for visualizing population snapshot computation graphs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from genes import NODE_TYPE_TO_INDEX

DECODED_GRAPH_DICT_KEY = "_decoded_graph_dict"
_INDEX_TO_NODE_TYPE: Dict[int, str] = {idx: name for name, idx in NODE_TYPE_TO_INDEX.items()}


@dataclass
class RenderContext:
    """Metadata shown on each graph (purely informational)."""

    generation: Optional[int] = None
    rank: Optional[int] = None
    task: Optional[str] = None


_DEF_NODE_COLORS = {
    "input": {"shape": "box", "fillcolor": "#E3F2FD"},
    "output": {"shape": "doubleoctagon", "fillcolor": "#FCE4EC"},
    "aten::": {"shape": "ellipse", "fillcolor": "#E8F5E9"},
    "prim::": {"shape": "ellipse", "fillcolor": "#F3E5F5"},
}


def load_population_snapshot(path: str | Path) -> Dict[str, Any]:
    """Load a serialized population snapshot from ``torch.save`` output."""

    snapshot = torch.load(Path(path), map_location="cpu")
    if not isinstance(snapshot, dict) or "entries" not in snapshot:
        raise ValueError("Malformed population snapshot: expected dict with 'entries'.")
    return snapshot


def find_latest_snapshot(directory: Path) -> Optional[Path]:
    """Return the most recently modified .pt snapshot in ``directory`` if any."""

    candidates = sorted(directory.glob("*.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def select_snapshot_entries(
    snapshot: Mapping[str, Any],
    *,
    top_k: Optional[int] = None,
    genome_ids: Sequence[int] | None = None,
    include_invalid: bool = False,
    sort_by: str = "fitness",
) -> List[Mapping[str, Any]]:
    """Filter and rank snapshot entries by fitness or genome id."""

    entries = list(snapshot.get("entries", []))
    filtered: List[Mapping[str, Any]] = []
    id_set = {int(gid) for gid in genome_ids} if genome_ids else None
    for entry in entries:
        if not include_invalid and entry.get("invalid_graph"):
            continue
        if id_set is not None and int(entry.get("genome_id", -1)) not in id_set:
            continue
        filtered.append(entry)

    def _fitness_key(item: Mapping[str, Any]) -> float:
        value = item.get("fitness")
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("-inf")

    if sort_by == "fitness":
        filtered.sort(key=_fitness_key, reverse=True)
    elif sort_by == "genome_id":
        filtered.sort(key=lambda item: int(item.get("genome_id", -1)))
    else:
        raise ValueError(f"Unsupported sort_by value: {sort_by!r}")

    if top_k is not None and top_k > 0:
        filtered = filtered[:top_k]
    return filtered


def attribute_key_to_name(attr_key: Any) -> str:
    """Best effort conversion of attribute keys to display-friendly names."""

    if hasattr(attr_key, "name"):
        return str(attr_key.name)
    if isinstance(attr_key, str):
        return attr_key
    return str(attr_key)


def _materialize_node_types(node_types: Any) -> List[int]:
    if node_types is None:
        return []
    if torch is not None and torch.is_tensor(node_types):
        flat = node_types.detach().cpu().view(-1).tolist()
    else:
        flat = list(node_types)
    result: List[int] = []
    for value in flat:
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            result.append(-1)
    return result


def _node_type_name(index: int) -> str:
    if not _INDEX_TO_NODE_TYPE and NODE_TYPE_TO_INDEX:
        _INDEX_TO_NODE_TYPE.update({idx: name for name, idx in NODE_TYPE_TO_INDEX.items()})
    return _INDEX_TO_NODE_TYPE.get(index, f"type_{index}")


def _normalize_node_attributes(node_attrs: Any, node_count: int) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    attrs_seq = list(node_attrs or [])
    for idx in range(node_count):
        attrs = attrs_seq[idx] if idx < len(attrs_seq) else {}
        normalized_dict: Dict[str, Any] = {}
        if isinstance(attrs, Mapping):
            for key, value in attrs.items():
                normalized_dict[attribute_key_to_name(key)] = value
        normalized.append(normalized_dict)
    return normalized


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return f"{value:.4g}"
    return str(value)


def _summarize_attr_value(value: Any, *, max_elements: int = 3, max_chars: int = 32) -> str:
    if torch is not None and torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.numel() == 1:
            return _format_scalar(tensor.item())
        flat = tensor.view(-1).tolist()
        preview = ", ".join(_format_scalar(v) for v in flat[:max_elements])
        if len(flat) > max_elements:
            preview += ", …"
        shape = "×".join(str(int(dim)) for dim in tensor.shape)
        text = f"tensor[{shape}] {preview}"
    elif isinstance(value, (list, tuple)):
        preview = ", ".join(_format_scalar(v) for v in list(value)[:max_elements])
        if len(value) > max_elements:
            preview += ", …"
        text = f"[{preview}]"
    elif isinstance(value, Mapping):
        items = []
        for idx, (key, val) in enumerate(value.items()):
            if idx >= max_elements:
                items.append("…")
                break
            items.append(f"{key}:{_format_scalar(val)}")
        text = "{" + ", ".join(items) + "}"
    else:
        text = _format_scalar(value)
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def _edge_list(edge_index: Any, node_count: int) -> List[Tuple[int, int]]:
    if edge_index is None:
        return []
    if torch is not None and torch.is_tensor(edge_index):
        tensor = edge_index.detach().cpu().long()
        if tensor.dim() == 1:
            if tensor.numel() % 2:
                return []
            tensor = tensor.view(2, -1)
        if tensor.dim() != 2 or tensor.size(0) != 2:
            return []
        pairs = tensor.t().tolist()
    else:
        seq = list(edge_index)
        if not seq:
            return []
        if len(seq) == 2 and all(hasattr(part, "__iter__") for part in seq):
            pairs = list(zip(seq[0], seq[1]))
        else:
            pairs = seq
    edges: List[Tuple[int, int]] = []
    for pair in pairs:
        try:
            src, dst = pair
        except (TypeError, ValueError):
            continue
        try:
            src_i = int(src)
            dst_i = int(dst)
        except (TypeError, ValueError):
            continue
        if 0 <= src_i < node_count and 0 <= dst_i < node_count:
            edges.append((src_i, dst_i))
    return edges


def _escape_label(text: str) -> str:
    escaped = text.replace("\\", "\\\\")
    escaped = escaped.replace('"', '\\"')
    return escaped


def _node_style(node_type_name: str) -> Dict[str, str]:
    lowered = node_type_name.lower()
    if lowered in _DEF_NODE_COLORS:
        style = _DEF_NODE_COLORS[lowered].copy()
    elif any(lowered.startswith(prefix) for prefix in ("aten::", "prim::")):
        key = "aten::" if lowered.startswith("aten::") else "prim::"
        style = _DEF_NODE_COLORS[key].copy()
    else:
        style = {"shape": "ellipse", "fillcolor": "#ECEFF1"}
    style.setdefault("style", "filled")
    return style


def _mermaid_escape(text: str) -> str:
    return text.replace('"', '\\"').replace("<", "&lt;").replace(">", "&gt;")


def build_mermaid_graph(
    entry: Mapping[str, Any],
    *,
    context: RenderContext | None = None,
    max_attr_lines: int | None = None,
    max_attr_value_chars: int = 32,
    rankdir: str = "LR",
    highlight_invalid: bool = True,
) -> str:
    graph_dict = entry.get("graph") or {}
    node_type_ids = _materialize_node_types(graph_dict.get("node_types"))
    node_count = len(node_type_ids)
    if node_count == 0:
        node_attrs = graph_dict.get("node_attributes") or []
        node_count = len(node_attrs)
        node_type_ids = list(range(node_count))
    nodes = [_node_type_name(idx) for idx in node_type_ids]
    attrs = _normalize_node_attributes(graph_dict.get("node_attributes"), len(nodes))
    edges = _edge_list(graph_dict.get("edge_index"), len(nodes))

    orientation = rankdir.upper()
    if orientation not in {"LR", "RL", "TB", "BT"}:
        orientation = "LR"
    ctx = context or RenderContext()
    label_parts = [f"genome={entry.get('genome_id')}"]
    if ctx.generation is not None:
        label_parts.append(f"gen={ctx.generation}")
    if ctx.rank is not None:
        label_parts.append(f"rank={ctx.rank}")
    if entry.get("fitness") is not None:
        try:
            label_parts.append(f"fitness={float(entry['fitness']):.4f}")
        except (TypeError, ValueError):
            label_parts.append(f"fitness={entry['fitness']}")
    if entry.get("species_id") is not None:
        label_parts.append(f"species={entry['species_id']}")
    if ctx.task:
        label_parts.append(f"task={ctx.task}")
    if highlight_invalid and entry.get("invalid_graph"):
        reason = entry.get("invalid_reason") or "invalid"
        label_parts.append(f"status=invalid({reason})")

    lines = [f"graph {orientation}"]
    if label_parts:
        lines.append("  %% " + " | ".join(label_parts))

    node_ids: List[str] = []
    for idx, (node_name, node_attrs) in enumerate(zip(nodes, attrs)):
        node_id = f"node_{idx}"
        node_ids.append(node_id)
        attr_lines: List[str] = [f"{idx}: {node_name}"]
        for key in sorted(node_attrs.keys()):
            if key == "node_type":
                continue
            summary = _summarize_attr_value(node_attrs[key], max_chars=max_attr_value_chars)
            attr_lines.append(f"{key}={summary}")
            if max_attr_lines is not None and len(attr_lines) - 1 >= max_attr_lines:
                break
        escaped_lines = [_mermaid_escape(line) for line in attr_lines]
        label = "<br/>".join(escaped_lines)
        lines.append(f'  {node_id}["{label}"]')

    for src, dst in edges:
        lines.append(f"  node_{src} --> node_{dst}")

    if not edges:
        lines.append("  %% Graph has no edges")

    if highlight_invalid and entry.get("invalid_graph") and node_ids:
        lines.append("  classDef invalid fill:#FFEBEE,stroke:#C62828,color:#C62828;")
        lines.append("  class " + ",".join(node_ids) + " invalid;")

    return "\n".join(lines)


def save_summary(
    entries: Sequence[Mapping[str, Any]], destination: Path, *, extra: Dict[str, Any] | None = None
) -> Path:
    payload = {
        "entries": [
            {
                "genome_id": entry.get("genome_id"),
                "species_id": entry.get("species_id"),
                "fitness": entry.get("fitness"),
                "invalid_graph": entry.get("invalid_graph"),
                "invalid_reason": entry.get("invalid_reason"),
            }
            for entry in entries
        ]
    }
    if extra:
        payload.update(extra)
    destination.write_text(json.dumps(payload, indent=2))
    return destination
