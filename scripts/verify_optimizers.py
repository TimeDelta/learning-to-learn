#!/usr/bin/env python3
"""Quick sanity check for TorchScript-style optimizers."""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Type

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

ROOT = Path(__file__).resolve().parents[1]
OPT_DIR = ROOT / "computation_graphs" / "optimizers"


def iter_optimizer_classes(module) -> Iterable[Type[nn.Module]]:
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if not isinstance(attr, type):
            continue
        if not issubclass(attr, nn.Module) or attr is nn.Module:
            continue
        # Only keep classes defined in this module
        if getattr(attr, "__module__", None) != module.__name__:
            continue
        yield attr


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def make_dataset(num_samples: int = 64, input_dim: int = 4, device: torch.device | None = None):
    torch.manual_seed(0)
    x = torch.randn(num_samples, input_dim, device=device)
    true_w = torch.randn(input_dim, 1, device=device)
    true_b = torch.randn(1, device=device)
    y = x @ true_w + true_b
    return x, y


def make_model(input_dim: int = 4, device: torch.device | None = None) -> nn.Module:
    model = nn.Linear(input_dim, 1)
    return model.to(device)


def apply_optimizer(
    optimizer: nn.Module,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    steps: int,
    tol: float,
) -> Tuple[bool, bool, List[float]]:
    loss_hist: List[float] = []
    prev_loss = torch.tensor(math.inf, device=device)
    any_param_change = False
    loss_improved = False

    named_params: List[Tuple[str, Parameter]] = list(model.named_parameters())

    for _ in range(steps):
        preds = model(inputs)
        loss = torch.mean((preds - targets) ** 2)
        loss_hist.append(loss.item())

        new_values: Dict[str, torch.Tensor] = optimizer(loss, prev_loss, named_params)
        if not isinstance(new_values, dict):
            raise RuntimeError("Optimizer must return Dict[str, Tensor]")

        for name, param in named_params:
            if name not in new_values:
                raise KeyError(f"Missing update for parameter {name}")
            updated = new_values[name].detach()
            if updated.shape != param.data.shape:
                raise RuntimeError(f"Shape mismatch for {name}: expected {param.data.shape}, got {updated.shape}")
            if not torch.allclose(updated, param.data):
                any_param_change = True
            param.data.copy_(updated)

        prev_loss = loss.detach()
        named_params = list(model.named_parameters())

    if loss_hist and min(loss_hist) < loss_hist[0] - tol:
        loss_improved = True

    return any_param_change, loss_improved, loss_hist


def main():
    parser = argparse.ArgumentParser(description="Verify optimizer modules")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device(args.device)
    inputs, targets = make_dataset(device=device)

    failures = []
    successes = []

    for path in sorted(OPT_DIR.glob("*.py")):
        module = load_module(path)
        classes = list(iter_optimizer_classes(module))
        if not classes:
            print(f"[SKIP] {path.name}: no Backprop nn.Module classes found")
            continue

        for cls in classes:
            optimizer = cls().to(device)
            model = make_model(device=device)
            try:
                changed, improved, losses = apply_optimizer(
                    optimizer, model, inputs, targets, device, args.steps, args.tol
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((path.name, cls.__name__, str(exc)))
                print(f"[FAIL] {path.name}:{cls.__name__} -> {exc}")
                continue

            if not changed:
                msg = "parameters never changed"
                failures.append((path.name, cls.__name__, msg))
                print(f"[FAIL] {path.name}:{cls.__name__} -> {msg}")
            elif not improved:
                msg = "loss did not decrease"
                failures.append((path.name, cls.__name__, msg))
                print(f"[WARN] {path.name}:{cls.__name__} -> {msg}")
            else:
                successes.append((path.name, cls.__name__, losses[0], min(losses)))
                print(f"[OK] {path.name}:{cls.__name__} loss {losses[0]:.4f} -> {min(losses):.4f}")

    print("\nSummary:")
    print(f"  {len(successes)} optimizers passed")
    print(f"  {len(failures)} optimizers failed")
    if failures:
        for name, cls_name, reason in failures:
            print(f"    - {name}:{cls_name} -> {reason}")
        sys.exit(1)


if __name__ == "__main__":
    if not OPT_DIR.exists():
        raise SystemExit(f"Optimizer directory not found: {OPT_DIR}")
    main()
