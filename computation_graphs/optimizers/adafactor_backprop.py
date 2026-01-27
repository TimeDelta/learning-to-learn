from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def _reshape_to_matrix(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() <= 1:
        raise ValueError("Tensor must be at least 2-D to reshape as matrix")
    first_dim = tensor.shape[0]
    return tensor.reshape(first_dim, -1)


class AdaFactorBackprop(nn.Module):
    """TorchScript-friendly AdaFactor variant."""

    exp_avg: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    exp_avg_sq: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    exp_avg_sq_row: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    exp_avg_sq_col: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(
        self,
        step_size: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-30,
        clip_threshold: float = 1.0,
    ):
        super().__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.clip_threshold = clip_threshold
        self.step = 0
        self.exp_avg = {}
        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}

    def _rms(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.mean(tensor * tensor) + self.eps)

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
            if name not in self.exp_avg:
                self.exp_avg[name] = torch.zeros_like(param)
                if param.dim() <= 1:
                    self.exp_avg_sq[name] = torch.zeros_like(param)
                else:
                    matrix = _reshape_to_matrix(param)
                    self.exp_avg_sq_row[name] = torch.zeros(matrix.size(0), device=param.device, dtype=param.dtype)
                    self.exp_avg_sq_col[name] = torch.zeros(matrix.size(1), device=param.device, dtype=param.dtype)
            elif param.dim() > 1:
                matrix = _reshape_to_matrix(param)
                row_state = self.exp_avg_sq_row[name]
                col_state = self.exp_avg_sq_col[name]
                if row_state.size(0) != matrix.size(0):
                    row_state = torch.zeros(matrix.size(0), device=param.device, dtype=param.dtype)
                if col_state.size(0) != matrix.size(1):
                    col_state = torch.zeros(matrix.size(1), device=param.device, dtype=param.dtype)
                self.exp_avg_sq_row[name] = row_state
                self.exp_avg_sq_col[name] = col_state

            exp_avg = self.exp_avg[name]
            exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad

            if param.dim() <= 1:
                exp_avg_sq = self.exp_avg_sq[name]
                exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * (grad * grad)
                denom = torch.sqrt(exp_avg_sq) + self.eps
                self.exp_avg_sq[name] = exp_avg_sq
            else:
                grad_matrix = _reshape_to_matrix(grad)
                row_state = self.exp_avg_sq_row[name]
                col_state = self.exp_avg_sq_col[name]
                grad_sq = grad_matrix * grad_matrix
                row_mean = grad_sq.mean(dim=1)
                col_mean = grad_sq.mean(dim=0)
                row_state = self.beta2 * row_state + (1 - self.beta2) * row_mean
                col_state = self.beta2 * col_state + (1 - self.beta2) * col_mean
                row_fact = (row_state / (row_state.mean() + self.eps)).sqrt().unsqueeze(1)
                col_fact = (col_state / (col_state.mean() + self.eps)).sqrt().unsqueeze(0)
                factored = row_fact * col_fact
                denom = factored.reshape_as(grad_matrix).reshape_as(param) + self.eps
                self.exp_avg_sq_row[name] = row_state
                self.exp_avg_sq_col[name] = col_state

            update = exp_avg / denom
            rms_update = self._rms(update)
            if self.clip_threshold > 0:
                clip_denom = torch.maximum(
                    torch.tensor(1.0, dtype=update.dtype, device=update.device),
                    rms_update / self.clip_threshold,
                )
                update = update / clip_denom

            new_params[name] = param - self.step_size * update
            self.exp_avg[name] = exp_avg

        return new_params


if __name__ == "__main__":
    optimizer = torch.jit.script(AdaFactorBackprop())
    torch.jit.save(optimizer, __file__.replace(".py", ".pt"))
    print(optimizer.graph)
