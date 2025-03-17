import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import warnings

from typing import Dict, List, Tuple

class BackpropGD(nn.Module):
    def __init__(self, step_size=0.1):
        """
        Backprop-Based Gradient Descent on a per-parameter basis.
        - step_size: learning rate multiplier.
        """
        super(BackpropGD, self).__init__()
        self.step_size = step_size

    def forward(self, loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]) -> Dict[str, torch.Tensor]:
        param_list = [param for _, param in named_parameters]
        # create_graph=False for efficiency
        grads = torch.autograd.grad([loss], param_list, create_graph=False)

        new_params = {}
        for (name, param), grad in zip(named_parameters, grads):
            if grad is None:
                warnings.warn("Gradient is zero for parameter " + name)
                grad = torch.zeros_like(param)
            new_params[name] = param - self.step_size * grad

        return new_params

if __name__ == "__main__":
    # generate dynamic computation graph:
    optimizer = torch.jit.script(BackpropGD())
    torch.jit.save(optimizer, __file__.replace('.py', '.pt'))
    print(optimizer.graph)
