# Generated by ChatGPT
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdagradBackprop(nn.Module):
    accum: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(self, step_size: float = 0.1, eps: float = 1e-8):
        super(AdagradBackprop, self).__init__()
        self.step_size = step_size
        self.eps = eps
        self.accum: Dict[str, torch.Tensor] = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [param for _, param in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}
        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            # Initialize accumulator for parameter if needed
            if name not in self.accum:
                self.accum[name] = torch.zeros_like(param)
            # Update squared-gradient accumulator
            self.accum[name] = self.accum[name] + grad * grad
            # Compute Adagrad update: scale by 1/sqrt(accum + eps)
            update = self.step_size * grad / (torch.sqrt(self.accum[name]) + self.eps)
            new_params[name] = param - update
        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdagradBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
