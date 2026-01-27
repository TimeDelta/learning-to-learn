from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LionBackprop(nn.Module):
    """Lion optimizer: momentum tracking + sign updates."""

    momentum: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-4,
        beta1: float = 0.9,
        beta2: float = 0.99,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.momentum = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            if name not in self.momentum:
                self.momentum[name] = torch.zeros_like(param)
            m = self.momentum[name]
            m = self.beta1 * m + (1 - self.beta1) * grad
            update = torch.sign(m)
            param_update = param - self.step_size * update
            m = self.beta2 * m + (1 - self.beta2) * grad
            new_params[name] = param_update
            self.momentum[name] = m

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(LionBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
