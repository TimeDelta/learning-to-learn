from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdaGradNormBackprop(nn.Module):
    """AdaGrad-Norm: global norm accumulator controls effective step size."""

    accum: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(self, step_size: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.step_size = step_size
        self.eps = eps
        self.accum = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            grad_norm_sq = torch.sum(grad * grad)
            if name not in self.accum:
                self.accum[name] = torch.zeros(1, device=param.device, dtype=param.dtype)
            accum = self.accum[name] + grad_norm_sq
            denom = torch.sqrt(accum) + self.eps
            step_scale = self.step_size / denom
            new_params[name] = param - step_scale * grad
            self.accum[name] = accum

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdaGradNormBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
