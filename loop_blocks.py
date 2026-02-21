"""Registry utilities for TorchScript block (e.g., prim::Loop body) payloads.

This file avoids any type of pre-defined templating system: it simply hashes the
raw nested block dictionaries that come out of TorchScript, stores them for later,
and exposes helpers to encode/decode those blocks as tensors so the VAE decoder can
emit the bodies directly. Downstream repair logic reuses the registry to fetch the
exact payload that the decoder already produced, ensuring nested control-flow blocks
survive round-trips without relying on pre-exported templates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import zlib
from typing import Any, Dict, Iterable, List, Mapping

import torch

_REGISTRY_LOCK = threading.Lock()
_BLOCK_REGISTRY: Dict[int, Dict[str, Any]] = {}
DEFAULT_BLOCK_PAYLOAD_VALUE_DIM = 2048


def _canonicalize(block_dict: Dict[str, Any]) -> str:
    return json.dumps(block_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _register_block_recursive(block_dict: Dict[str, Any]) -> int:
    serialized = _canonicalize(block_dict)
    digest = hashlib.sha256(serialized.encode("utf-8")).digest()
    block_id = int.from_bytes(digest[:8], byteorder="big", signed=False)
    with _REGISTRY_LOCK:
        if block_id not in _BLOCK_REGISTRY:
            _BLOCK_REGISTRY[block_id] = copy.deepcopy(block_dict)
    # ensure nested blocks also registered
    for node in block_dict.get("nodes", []) or []:
        for inner in node.get("blocks", []) or []:
            _register_block_recursive(inner)
    return block_id


def register_graph_blocks(graph_ir: Mapping[str, Any]) -> Dict[int, List[int]]:
    """Register all nested blocks within graph_ir and return per-node block ids."""

    node_block_ids: Dict[int, List[int]] = {}
    for idx, node in enumerate(graph_ir.get("nodes", []) or []):
        blocks = []
        for inner in node.get("blocks", []) or []:
            blocks.append(_register_block_recursive(inner))
        if blocks:
            node_block_ids[idx] = blocks
    return node_block_ids


def snapshot_registry(block_ids: Iterable[int]) -> Dict[int, Dict[str, Any]]:
    """Return a copy of registry entries for the provided ids."""

    subset: Dict[int, Dict[str, Any]] = {}
    for block_id in block_ids:
        block = _BLOCK_REGISTRY.get(block_id)
        if block is not None:
            subset[block_id] = copy.deepcopy(block)
    return subset


def prime_registry(block_registry: Mapping[int, Dict[str, Any]] | None) -> None:
    """Seed the global registry with pre-serialized block payloads."""

    if not block_registry:
        return
    for block_id, payload in block_registry.items():
        _register_block_recursive(payload)


def get_block(block_id: int) -> Dict[str, Any]:
    block = _BLOCK_REGISTRY.get(block_id)
    if block is None:
        raise KeyError(f"Unknown loop block id {block_id}")
    return block


def registered_block_ids() -> List[int]:
    with _REGISTRY_LOCK:
        return list(_BLOCK_REGISTRY.keys())


def encode_block_payload(
    block_dict: Dict[str, Any],
    *,
    max_value_dim: int = DEFAULT_BLOCK_PAYLOAD_VALUE_DIM,
) -> torch.Tensor:
    """Serialize a block payload into a 1-D tensor for decoder training."""

    if max_value_dim <= 1:
        raise ValueError("max_value_dim must be > 1 to encode block payloads")
    canonical = _canonicalize(block_dict)
    compressed = zlib.compress(canonical.encode("utf-8"))
    max_bytes = max_value_dim - 1
    if len(compressed) > max_bytes:
        raise ValueError(f"Block payload exceeds capacity ({len(compressed)} bytes > {max_bytes} limit).")
    tensor = torch.zeros(len(compressed) + 1, dtype=torch.float32)
    tensor[0] = float(len(compressed))
    if compressed:
        tensor[1:] = torch.tensor(list(compressed), dtype=torch.float32)
    return tensor


def decode_block_payload(value: Any) -> Dict[str, Any]:
    """Invert encode_block_payload, returning the original block dictionary."""

    if value is None:
        raise ValueError("Cannot decode empty block payload")
    tensor = torch.as_tensor(value, dtype=torch.float32).view(-1)
    if tensor.numel() < 1:
        raise ValueError("Encoded block payload tensor is empty")
    length = int(round(float(tensor[0].item())))
    if length < 0:
        raise ValueError(f"Invalid block payload length {length}")
    if tensor.numel() < length + 1:
        raise ValueError("Encoded block payload truncated")
    raw_bytes = bytearray()
    for byte_val in tensor[1 : 1 + length]:
        clamped = max(0, min(255, int(round(float(byte_val.item())))))
        raw_bytes.append(clamped)
    try:
        decompressed = zlib.decompress(bytes(raw_bytes)) if raw_bytes else b""
    except zlib.error as exc:  # pragma: no cover - robustness for corrupt payloads
        raise ValueError(f"Failed to decompress block payload: {exc}") from exc
    if not decompressed:
        return {}
    try:
        return json.loads(decompressed.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in decoded block payload: {exc}") from exc


def register_block_payload_tensor(
    value: Any,
) -> tuple[int, Dict[str, Any]]:
    """Decode a payload tensor, register it, and return (block_id, payload)."""

    payload = decode_block_payload(value)
    block_id = _register_block_recursive(payload)
    return block_id, payload
