from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class IRPropPlusBackprop(nn.Module):
    """Improved Rprop+ with weight backtracking."""

    step_size: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    prev_grad: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    prev_update: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_init: float = 1e-2,
        step_min: float = 1e-6,
        step_max: float = 50.0,
        eta_plus: float = 1.2,
        eta_minus: float = 0.5,
    ):
        super().__init__()
        self.step_init = step_init
        self.step_min = step_min
        self.step_max = step_max
        self.eta_plus = eta_plus
        self.eta_minus = eta_minus
        self.step_size = {}
        self.prev_grad = {}
        self.prev_update = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False, allow_unused=True)
        new_params: Dict[str, torch.Tensor] = {}
        loss_increase = loss.item() > prev_loss.item() if prev_loss is not None else False

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if name not in self.step_size:
                self.step_size[name] = torch.full_like(param, self.step_init)
                self.prev_grad[name] = torch.zeros_like(param)
                self.prev_update[name] = torch.zeros_like(param)
            step = self.step_size[name]
            prev_grad = self.prev_grad[name]
            prev_update = self.prev_update[name]
            sign = grad * prev_grad
            step = torch.where(
                sign > 0,
                torch.clamp(step * self.eta_plus, max=self.step_max),
                torch.where(sign < 0, torch.clamp(step * self.eta_minus, min=self.step_min), step),
            )
            grad_eff = torch.sign(grad)
            neg_mask = sign < 0
            grad_eff = torch.where(neg_mask, torch.zeros_like(grad_eff), grad_eff)
            update = -step * grad_eff
            if loss_increase:
                update = torch.where(neg_mask, -prev_update, update)
            else:
                update = torch.where(neg_mask, torch.zeros_like(update), update)
            new_param = param + update
            new_params[name] = new_param
            self.step_size[name] = step
            self.prev_grad[name] = torch.where(neg_mask, torch.zeros_like(grad), grad)
            self.prev_update[name] = update

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(IRPropPlusBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
