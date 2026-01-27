from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SGDWBackprop(nn.Module):
    """Momentum SGD with decoupled weight decay (SGD-W)."""

    velocity: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        nesterov: bool = False,
    ):
        super().__init__()
        self.step_size = step_size
        self.momentum_coeff = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.velocity = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if name not in self.velocity:
                self.velocity[name] = torch.zeros_like(param)
            velocity = self.momentum_coeff * self.velocity[name] + grad
            if self.nesterov:
                update = grad + self.momentum_coeff * velocity
            else:
                update = velocity
            decayed = param * (1 - self.step_size * self.weight_decay)
            new_params[name] = decayed - self.step_size * update
            self.velocity[name] = velocity

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(SGDWBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
