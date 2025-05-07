import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from typing import Dict, List, Tuple

class AdaBeliefBackprop(nn.Module):
    moment1: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])
    moment2: Dict[str, torch.Tensor] = torch.jit.Attribute({}, Dict[str, torch.Tensor])

    def __init__(self, step_size: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super(AdaBeliefBackprop, self).__init__()
        self.step_size = step_size
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.step = 0
        self.moment1: Dict[str, torch.Tensor] = {}
        self.moment2: Dict[str, torch.Tensor] = {}

    def forward(self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]) -> Dict[str, torch.Tensor]:
        self.step += 1
        params = [param for _, param in named_parameters]
        grads = torch.autograd.grad([loss], params, create_graph=False)

        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                grad = torch.zeros_like(param)

            # Initialize moments if needed
            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.moment2[name] = torch.zeros_like(param)

            m = self.moment1[name]
            s = self.moment2[name]

            # Update first moment
            m = self.beta1 * m + (1 - self.beta1) * grad

            # Update second moment on gradient *deviation*
            diff = grad - m
            s = self.beta2 * s + (1 - self.beta2) * (diff * diff)

            # Bias-correct moments
            m_hat = m / (1 - self.beta1 ** self.step)
            s_hat = s / (1 - self.beta2 ** self.step)

            # AdaBelief update
            new_params[name] = param - self.step_size * (m_hat / (torch.sqrt(s_hat) + self.eps))

            self.moment1[name] = m
            self.moment2[name] = s

        return new_params

if __name__ == "__main__":
    optimizer = torch.jit.script(AdaBeliefBackprop())
    torch.jit.save(optimizer, __file__.replace('.py', '.pt'))
    print(optimizer.graph)
