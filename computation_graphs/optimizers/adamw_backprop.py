from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdamWBackprop(nn.Module):
    """Adam with decoupled weight decay."""

    moment1: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    moment2: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.step = 0
        self.moment1 = {}
        self.moment2 = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        beta1_pow = self.beta1**self.step
        beta2_pow = self.beta2**self.step
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.moment2[name] = torch.zeros_like(param)
            m = self.beta1 * self.moment1[name] + (1 - self.beta1) * grad
            v = self.beta2 * self.moment2[name] + (1 - self.beta2) * (grad * grad)
            m_hat = m / (1 - beta1_pow)
            v_hat = v / (1 - beta2_pow)
            decayed = param - self.step_size * self.weight_decay * param
            new_params[name] = decayed - self.step_size * m_hat / (torch.sqrt(v_hat) + self.eps)
            self.moment1[name] = m
            self.moment2[name] = v

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdamWBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
