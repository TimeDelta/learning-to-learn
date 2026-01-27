from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class FTRLProximalBackprop(nn.Module):
    """Follow-the-Regularized-Leader (proximal) optimizer."""

    accumulator: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    z_buffer: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        lambda1: float = 1e-4,
        lambda2: float = 1e-4,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.accumulator = {}
        self.z_buffer = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if name not in self.accumulator:
                self.accumulator[name] = torch.zeros_like(param)
                self.z_buffer[name] = torch.zeros_like(param)
            acc = self.accumulator[name]
            z = self.z_buffer[name]
            new_acc = acc + grad * grad
            sigma = (torch.sqrt(new_acc) - torch.sqrt(acc)) / self.alpha
            z = z + grad - sigma * param
            l1 = self.lambda1
            mask = torch.abs(z) > l1
            denom = (self.beta + torch.sqrt(new_acc)) / self.alpha + self.lambda2
            new_param = torch.zeros_like(param)
            new_param[mask] = -((z[mask] - torch.sign(z[mask]) * l1) / denom[mask])
            new_params[name] = new_param
            self.accumulator[name] = new_acc
            self.z_buffer[name] = z

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(FTRLProximalBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
