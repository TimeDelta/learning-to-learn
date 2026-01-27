from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class RMSPropNormBackprop(nn.Module):
    """RMSProp variant that normalizes by a global gradient norm EMA."""

    norm_avg: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-2,
        beta: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.norm_avg = {}

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
            grad_norm = torch.sqrt(torch.mean(grad * grad) + self.eps).detach()
            if name not in self.norm_avg:
                self.norm_avg[name] = grad_norm
            avg = self.beta * self.norm_avg[name] + (1 - self.beta) * grad_norm
            self.norm_avg[name] = avg
            normalized_grad = grad / (avg + self.eps)
            new_params[name] = param - self.step_size * normalized_grad

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(RMSPropNormBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
