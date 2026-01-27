from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdanBackprop(nn.Module):
    """Implementation of the Adan optimizer (adaptive Nesterov momentum)."""

    exp_avg: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    exp_avg_diff: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    exp_avg_sq: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    prev_grad: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-3,
        beta1: float = 0.98,
        beta2: float = 0.92,
        beta3: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.eps = eps
        self.weight_decay = weight_decay
        self.step = 0
        self.exp_avg = {}
        self.exp_avg_diff = {}
        self.exp_avg_sq = {}
        self.prev_grad = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            if name not in self.exp_avg:
                self.exp_avg[name] = torch.zeros_like(param)
                self.exp_avg_diff[name] = torch.zeros_like(param)
                self.exp_avg_sq[name] = torch.zeros_like(param)
                self.prev_grad[name] = torch.zeros_like(param)

            prev_g = self.prev_grad[name]
            grad_diff = grad - prev_g
            exp_avg = self.beta1 * self.exp_avg[name] + (1 - self.beta1) * grad
            exp_avg_diff = self.beta2 * self.exp_avg_diff[name] + (1 - self.beta2) * grad_diff
            merged = grad + self.beta2 * exp_avg_diff
            exp_avg_sq = self.beta3 * self.exp_avg_sq[name] + (1 - self.beta3) * (merged * merged)

            denom = torch.sqrt(exp_avg_sq) + self.eps
            update = (exp_avg + self.beta2 * exp_avg_diff) / denom

            new_params[name] = param - self.step_size * update

            self.exp_avg[name] = exp_avg
            self.exp_avg_diff[name] = exp_avg_diff
            self.exp_avg_sq[name] = exp_avg_sq
            self.prev_grad[name] = grad

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdanBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
