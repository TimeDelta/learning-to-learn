import torch
import torch.nn as nn
from typing import Dict, Callable

class AdamOpt(nn.Module):
    def __init__(self, step_size: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Basic Adam optimizer that computes parameter updates using Finite Diff.

        Args:
            step_size: The learning rate (alpha).
            beta1: Exponential decay rate for the first moment estimates.
            beta2: Exponential decay rate for the second moment estimates.
            eps: A small constant to avoid division by zero.
        """
        super(AdamOpt, self).__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.moment1: Dict[str, torch.Tensor] = {}
        self.moment2: Dict[str, torch.Tensor] = {}

    def forward(self, model: nn.Module, loss_fn: Callable[[nn.Module], torch.Tensor]) -> nn.Module:
        new_params: Dict[str, torch.Tensor] = {}
        self.step += 1
        for name, param in model.named_parameters():
            original = param.clone()

            with torch.no_grad():
                param.copy_(original + self.finite_diff_eps)
            loss_plus = loss_fn(model)

            with torch.no_grad():
                param.copy_(original - self.finite_diff_eps)
            loss_minus = loss_fn(model)

            with torch.no_grad():
                param.copy_(original)

            grad_est = (loss_plus - loss_minus) / (2 * self.finite_diff_eps)

            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.moment2[name] = torch.zeros_like(param)
            m1 = self.moment1[name]
            m2 = self.moment2[name]

            new_m1 = self.beta1 * m1 + (1 - self.beta1) * grad_est
            new_m2 = self.beta2 * m2 + (1 - self.beta2) * (grad_est * grad_est)
            bias_coirrected_m1 = new_m1 / (1 - self.beta1 ** self.step)
            bias_correceted_m2 = new_m2 / (1 - self.beta2 ** self.step)

            updated_param = param - self.step_size * m_hat / (torch.sqrt(v_hat) + self.eps)
            new_params[name] = updated_param

            self.moment1[name] = new_m
            self.moment2[name] = new_v

        model.load_state_dict(new_params)
        return model

if __name__ == "__main__":
    optimizer = torch.jit.script(AdamOptimizer())
    torch.jit.save(optimizer, __file__.replace('.py', '.pt'))
    print(optimizer.graph)
