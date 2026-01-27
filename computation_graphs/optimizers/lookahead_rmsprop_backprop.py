from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class LookaheadRMSPropBackprop(nn.Module):
    """RMSProp optimizer wrapped with Lookahead slow/fast dynamics."""

    avg_sq: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    momentum: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    slow_params: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-3,
        beta: float = 0.99,
        momentum: float = 0.0,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        sync_period: int = 5,
        slow_step: float = 0.5,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta = beta
        self.momentum_coeff = momentum
        self.eps = eps
        self.weight_decay = weight_decay
        self.sync_period = max(1, sync_period)
        self.slow_step = slow_step
        self.step = 0
        self.avg_sq = {}
        self.momentum = {}
        self.slow_params = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        sync = self.step % self.sync_period == 0
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param
            if name not in self.avg_sq:
                self.avg_sq[name] = torch.zeros_like(param)
                self.momentum[name] = torch.zeros_like(param)
                self.slow_params[name] = param.detach().clone()

            avg_sq = self.beta * self.avg_sq[name] + (1 - self.beta) * (grad * grad)
            denom = torch.sqrt(avg_sq) + self.eps
            update = grad / denom
            momentum_buf = self.momentum[name]
            if self.momentum_coeff > 0:
                momentum_buf = self.momentum_coeff * momentum_buf + self.step_size * update
                fast_param = param - momentum_buf
            else:
                fast_param = param - self.step_size * update
            slow = self.slow_params[name]
            if sync:
                slow = slow + self.slow_step * (fast_param - slow)
                fast_param = slow.clone()
            new_params[name] = fast_param

            self.avg_sq[name] = avg_sq
            self.momentum[name] = momentum_buf
            self.slow_params[name] = slow

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(LookaheadRMSPropBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
