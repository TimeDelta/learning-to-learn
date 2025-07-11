# Generated by ChatGPT
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class QHAdamBackprop(nn.Module):
    moment1: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    moment2: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        nu1: float = 1.0,
        nu2: float = 1.0,
        eps: float = 1e-8,
    ):
        super(QHAdamBackprop, self).__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.nu1 = nu1
        self.nu2 = nu2
        self.eps = eps
        self.step = 0
        self.moment1: Dict[str, torch.Tensor] = {}
        self.moment2: Dict[str, torch.Tensor] = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [param for _, param in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)  # Initialize moments if needed
            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.moment2[name] = torch.zeros_like(param)
            m = self.moment1[name]
            v = self.moment2[name]  # Adam-style updates of moments
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)  # Compute quasi-hyperbolic numerator and denominator
            num = (1 - self.nu1) * grad + self.nu1 * m
            den = torch.sqrt((1 - self.nu2) * (grad * grad) + self.nu2 * v) + self.eps
            new_params[name] = param - self.step_size * num / den
            self.moment1[name] = m
            self.moment2[name] = v
        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(QHAdamBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
