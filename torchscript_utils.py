import io
from typing import BinaryIO

import torch


def serialize_script_module(module: torch.jit.ScriptModule) -> bytes:
    """Serialize a TorchScript module into raw bytes."""
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    return buffer.getvalue()


def load_script_module(blob: bytes | bytearray | memoryview | BinaryIO) -> torch.jit.ScriptModule:
    """Load a TorchScript module from raw bytes or a file-like object."""
    if isinstance(blob, (bytes, bytearray, memoryview)):
        buffer: BinaryIO = io.BytesIO(blob)
    else:
        buffer = blob
    buffer.seek(0)
    return torch.jit.load(buffer)
