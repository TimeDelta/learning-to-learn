from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LookaheadAdamBackprop(nn.Module):
    """Adam optimizer wrapped with Lookahead slow/fast weights."""

    moment1: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    moment2: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    slow_params: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        sync_period: int = 5,
        slow_step: float = 0.5,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.sync_period = max(1, sync_period)
        self.slow_step = slow_step
        self.step = 0
        self.moment1 = {}
        self.moment2 = {}
        self.slow_params = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        beta1_pow = self.beta1**self.step
        beta2_pow = self.beta2**self.step
        new_params: Dict[str, torch.Tensor] = {}
        sync = self.step % self.sync_period == 0

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.moment2[name] = torch.zeros_like(param)
                self.slow_params[name] = param.detach().clone()

            m = self.moment1[name]
            v = self.moment2[name]
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            m_hat = m / (1 - beta1_pow)
            v_hat = v / (1 - beta2_pow)
            fast_param = param - self.step_size * m_hat / (torch.sqrt(v_hat) + self.eps)

            slow = self.slow_params[name]
            if sync:
                slow = slow + self.slow_step * (fast_param - slow)
                fast_param = slow.clone()
            new_params[name] = fast_param

            self.moment1[name] = m
            self.moment2[name] = v
            self.slow_params[name] = slow

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(LookaheadAdamBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
