from __future__ import annotations

import ast
import functools
import inspect
import io
import weakref
from typing import Any, BinaryIO, Dict, List, Tuple

import torch

_NAMED_PARAM_KEYS = ("named_parameters", "named_params")
_EXPECTED_PARAM_TYPE_SRC = "List[Tuple[str, torch.Tensor]]"
_EXPECTED_RETURN_TYPE_SRC = "Dict[str, torch.Tensor]"
_EXPECTED_PARAM_TYPE = List[Tuple[str, torch.Tensor]]
_EXPECTED_RETURN_TYPE = Dict[str, torch.Tensor]


class _ForwardPatchInfo:
    __slots__ = ("param_names", "ensure_return", "filename", "first_lineno")

    def __init__(self, filename: str | None, first_lineno: int | None) -> None:
        self.param_names: set[str] = set()
        self.ensure_return: bool = False
        self.filename = filename
        self.first_lineno = first_lineno


_FORWARD_PATCHES: "weakref.WeakKeyDictionary[Any, _ForwardPatchInfo]" = weakref.WeakKeyDictionary()
_PATCHES_BY_LOCATION: Dict[tuple[str | None, int | None], _ForwardPatchInfo] = {}


def _extract_forward_function(target: Any) -> Any:
    """Return the underlying Python function object for forward if available."""

    if target is None:
        return None
    if inspect.isclass(target):
        func = target.__dict__.get("forward")
        return func
    forward = getattr(target, "forward", None)
    if forward is None:
        return None
    # Bound methods carry the actual function on __func__.
    return getattr(forward, "__func__", forward)


def _ensure_named_parameter_annotations(target: Any) -> None:
    """Record optimizer-style signatures so TorchScript can infer types."""

    forward = _extract_forward_function(target)
    if forward is None or not callable(forward):
        return

    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        return

    annotations = getattr(forward, "__annotations__", None)
    if annotations is None:
        annotations = {}
        forward.__annotations__ = annotations

    missing_params: list[str] = []
    for key in _NAMED_PARAM_KEYS:
        if key not in parameters:
            continue
        current = annotations.get(key)
        if current in (None, torch.Tensor):
            annotations[key] = _EXPECTED_PARAM_TYPE
            missing_params.append(key)

    ret = annotations.get("return")
    needs_return = ret in (None, torch.Tensor)
    if needs_return:
        annotations["return"] = _EXPECTED_RETURN_TYPE

    if not missing_params and not needs_return:
        return

    patch = _FORWARD_PATCHES.get(forward)
    if patch is None:
        code = getattr(forward, "__code__", None)
        filename = getattr(code, "co_filename", None)
        first_lineno = getattr(code, "co_firstlineno", None)
        patch = _ForwardPatchInfo(filename, first_lineno)
        _FORWARD_PATCHES[forward] = patch
        _PATCHES_BY_LOCATION[(filename, first_lineno)] = patch
    patch.param_names.update(missing_params)
    patch.ensure_return |= needs_return

    globals_map = getattr(forward, "__globals__", None)
    if isinstance(globals_map, dict):
        globals_map.setdefault("List", List)
        globals_map.setdefault("Tuple", Tuple)
        globals_map.setdefault("Dict", Dict)


def _maybe_patch_optimizer_annotations(obj: Any) -> None:
    """Patch annotations on modules or module classes before scripting."""

    if inspect.isclass(obj) and issubclass(obj, torch.nn.Module):
        _ensure_named_parameter_annotations(obj)
        return
    if isinstance(obj, torch.nn.Module):
        _ensure_named_parameter_annotations(obj.__class__)


_ORIGINAL_TORCHSCRIPT = torch.jit.script


@functools.wraps(_ORIGINAL_TORCHSCRIPT)
def _script_with_optimizer_support(obj: Any, *args: Any, **kwargs: Any):
    """Ensure optimizer modules stay scriptable even without explicit typing."""

    try:
        _maybe_patch_optimizer_annotations(obj)
    except Exception:
        # Never block scripting when the helper fails—PyTorch will raise later
        # with its own diagnostics, ensuring we do not mask genuine issues.
        pass
    return _ORIGINAL_TORCHSCRIPT(obj, *args, **kwargs)


if not getattr(torch.jit.script, "_ltl_optimizer_patch", False):
    _script_with_optimizer_support._ltl_optimizer_patch = True
    torch.jit.script = _script_with_optimizer_support


_ORIGINAL_PARSE_DEF = torch.jit.frontend.parse_def


def _clone_annotation(expr_src: str, template: ast.AST) -> ast.AST:
    expr = ast.parse(expr_src, mode="eval").body
    return ast.copy_location(ast.fix_missing_locations(expr), template)


def _parse_def_with_optimizer_support(fn: Any):
    parsed = _ORIGINAL_PARSE_DEF(fn)
    patch = _FORWARD_PATCHES.get(fn)
    if patch is None:
        filename = getattr(getattr(fn, "__code__", None), "co_filename", None)
        first_lineno = getattr(getattr(fn, "__code__", None), "co_firstlineno", None)
        patch = _PATCHES_BY_LOCATION.get((filename, first_lineno))
    if not patch:
        return parsed

    fn_def = parsed.ast.body[0]

    args_obj = getattr(fn_def, "args", None)
    if args_obj is not None:
        for arg in getattr(args_obj, "args", []) + getattr(args_obj, "kwonlyargs", []):
            if not isinstance(arg, ast.arg):
                continue
            if arg.arg not in patch.param_names:
                continue
            if getattr(arg, "annotation", None) is None:
                arg.annotation = _clone_annotation(_EXPECTED_PARAM_TYPE_SRC, arg)

    if patch.ensure_return and getattr(fn_def, "returns", None) is None:
        fn_def.returns = _clone_annotation(_EXPECTED_RETURN_TYPE_SRC, fn_def)

    return parsed


if not getattr(torch.jit.frontend.parse_def, "_ltl_optimizer_patch", False):
    _parse_def_with_optimizer_support._ltl_optimizer_patch = True
    torch.jit.frontend.parse_def = _parse_def_with_optimizer_support


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
