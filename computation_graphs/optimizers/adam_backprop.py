from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class AdamBackprop(nn.Module):
    moment1: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    moment2: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(self, step_size: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super(AdamBackprop, self).__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.moment1: Dict[str, torch.Tensor] = {}
        self.moment2: Dict[str, torch.Tensor] = {}

    def forward(
        self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]
    ) -> Dict[str, torch.Tensor]:
        param_list = [param for _, param in named_parameters]
        # create_graph=False for efficiency
        grads = torch.autograd.grad([loss], param_list, create_graph=False, allow_unused=True)
        self.step += 1
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)
            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.moment2[name] = torch.zeros_like(param)
            moment1 = self.moment1[name]
            moment2 = self.moment2[name]

            beta1_tensor = torch.tensor(self.beta1, dtype=moment1.dtype, device=moment1.device)
            one_minus_beta1 = torch.tensor(1 - self.beta1, dtype=grad.dtype, device=grad.device)
            beta2_tensor = torch.tensor(self.beta2, dtype=moment2.dtype, device=moment2.device)
            one_minus_beta2 = torch.tensor(1 - self.beta2, dtype=grad.dtype, device=grad.device)

            new_moment1 = self.beta1 * moment1 + one_minus_beta1 * grad
            new_moment2 = self.beta2 * moment2 + one_minus_beta2 * (grad * grad)

            bias_corrected_moment1 = new_moment1 / (1 - self.beta1**self.step)
            bias_corrected_moment2 = new_moment2 / (1 - self.beta2**self.step)

            new_param = param - self.step_size * bias_corrected_moment1 / (
                torch.sqrt(bias_corrected_moment2) + self.eps
            )
            new_params[name] = new_param

            self.moment1[name] = new_moment1
            self.moment2[name] = new_moment2

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdamBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
