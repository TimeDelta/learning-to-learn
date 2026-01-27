from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdaNormBackprop(nn.Module):
    """Implements AdaNorm: normalize gradients to a running norm target."""

    norm_ema: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(self, step_size: float = 1e-2, beta: float = 0.95, eps: float = 1e-8):
        super().__init__()
        self.step_size = step_size
        self.beta = beta
        self.eps = eps
        self.norm_ema = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            grad_norm = torch.linalg.norm(grad).detach()
            if name not in self.norm_ema:
                self.norm_ema[name] = torch.ones_like(grad_norm)
            ema = self.beta * self.norm_ema[name] + (1 - self.beta) * grad_norm
            self.norm_ema[name] = ema
            scaled_grad = grad
            if grad_norm > 0:
                scaled_grad = grad / (grad_norm + self.eps) * ema
            new_params[name] = param - self.step_size * scaled_grad

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdaNormBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
