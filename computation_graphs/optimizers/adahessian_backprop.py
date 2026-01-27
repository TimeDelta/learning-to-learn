from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdaHessianBackprop(nn.Module):
    """AdaHessian: uses Hutchinson Hessian-diagonal estimates for curvature."""

    momentum: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-2,
        beta: float = 0.9,
        eps: float = 1e-8,
        hessian_power: float = 0.25,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta = beta
        self.eps = eps
        self.hessian_power = hessian_power
        self.momentum = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=True, retain_graph=True, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if name not in self.momentum:
                self.momentum[name] = torch.zeros_like(param)
            momentum = self.beta * self.momentum[name] + (1 - self.beta) * grad

            rademacher_dist = torch.randint(0, 2, param.shape, device=param.device, dtype=torch.int64)
            rademacher_dist = rademacher_dist.to(param.dtype) * 2 - 1
            grad_dot = torch.sum(grad * rademacher_dist)
            hvp = torch.autograd.grad(grad_dot, param, retain_graph=True, allow_unused=True)[0]
            if hvp is None:
                hvp = torch.zeros_like(param)
            diag_estimate = hvp * rademacher_dist
            denom = torch.pow(torch.abs(diag_estimate) + self.eps, self.hessian_power) + self.eps
            update = momentum / denom
            new_params[name] = param - self.step_size * update
            self.momentum[name] = momentum.detach()

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdaHessianBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
