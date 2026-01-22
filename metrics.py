from typing import Literal, Sequence

import torch
import torch.nn as nn


class Metric:
    name: str
    objective: Literal["min", "max"]
    # Numeric target viewed as "perfect" performance for this metric.
    best_value: float = 0.0

    @classmethod
    def canonicalize(cls, value: float) -> float:
        """Return value expressed as distance from the metric's best_value."""
        return float(value) - float(getattr(cls, "best_value", 0.0))


class MSELoss(Metric):
    name = "MSE"
    objective = "min"
    best_value = 0.0

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_tensor = torch.as_tensor(target, dtype=output.dtype, device=output.device)
        return nn.functional.mse_loss(output, target_tensor)


class AreaUnderTaskMetrics(Metric):
    name = "Area Under Task Metrics"
    objective = "min"
    best_value = 0.0


class TimeCost(Metric):
    name = "Time Cost"
    objective = "min"
    best_value = 0.0


class MemoryCost(Metric):
    name = "Memory Cost"
    objective = "min"
    best_value = 0.0


def metric_best_value(metric: Metric) -> float:
    """Return the numeric best_value for a metric instance or class."""
    return float(getattr(metric, "best_value", 0.0))


def sort_metrics_by_name(metrics: Sequence[Metric]):
    """Return a list of metrics sorted deterministically by their name."""
    return sorted(metrics, key=lambda metric: metric.name)
