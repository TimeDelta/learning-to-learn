from typing import Literal

import torch
import torch.nn as nn


class Metric:
    name: str
    objective: Literal["min", "max"]


class MSELoss(Metric):
    name = "MSE"
    objective = "min"

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_tensor = torch.as_tensor(target, dtype=output.dtype, device=output.device)
        return nn.functional.mse_loss(output, target_tensor)


class AreaUnderTaskMetrics(Metric):
    name = "Area Under Task Metrics"
    objective = "min"


class TimeCost(Metric):
    name = "Time Cost"
    objective = "min"


class MemoryCost(Metric):
    name = "Memory Cost"
    objective = "min"
