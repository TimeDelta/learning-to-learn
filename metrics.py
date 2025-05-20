import torch
import torch.nn as nn

from typing import Literal

class Metric:
    name: str
    objective: Literal['min', 'max']

class MSELoss(Metric):
    name = 'MSE'
    objective = 'min'

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(output, torch.tensor(target))

class AreaUnderTaskMetrics(Metric):
    name = 'Area Under Task Metrics'
    objective = 'min'

class TimeCost(Metric):
    name = 'Time Cost'
    objective = 'min'

class MemoryCost(Metric):
    name = 'Memory Cost'
    objective = 'min'
