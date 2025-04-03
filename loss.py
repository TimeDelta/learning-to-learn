import torch
import torch.nn as nn

class MSELoss:
    def __init__(self, input: torch.Tensor, target: torch.Tensor):
        self.input = input
        self.target = target
    def __call__(self, model: torch.nn.Module) -> torch.Tensor:
        output = model(self.input)
        return nn.functional.mse_loss(output, self.target)
