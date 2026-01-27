from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class ScaledSignSGDBackprop(nn.Module):
    """SignSGD with EMA magnitude scaling to mimic majority voting."""

    mag_ema: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(self, step_size: float = 1e-3, beta: float = 0.9, eps: float = 1e-8):
        super().__init__()
        self.step_size = step_size
        self.beta = beta
        self.eps = eps
        self.step = 0
        self.mag_ema = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}
        bias_correction = 1 - self.beta**self.step

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            magnitude = torch.mean(torch.abs(grad)).detach()
            if name not in self.mag_ema:
                self.mag_ema[name] = magnitude
            ema = self.beta * self.mag_ema[name] + (1 - self.beta) * magnitude
            self.mag_ema[name] = ema
            scaled_mag = ema / (bias_correction + self.eps)
            update = torch.sign(grad) * scaled_mag
            new_params[name] = param - self.step_size * update

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(ScaledSignSGDBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
