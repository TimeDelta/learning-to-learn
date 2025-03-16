import numpy as np
import torch
import torch.nn as nn

class MSELossFunctionWrapper(nn.Module):
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor):
        super(LossFunctionWrapper, self).__init__()
        self.inputs = inputs
        self.targets = targets

    def forward(self, model: nn.Module) -> torch.Tensor:
        output = model(self.inputs)
        return nn.functional.mse_loss(output, self.targets)
