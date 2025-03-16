import torch
import torch.nn as nn
from typing import Dict, Callable

class AdamOpt(nn.Module):
    def __init__(self, step_size: float = 0.1, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        """
        Basic Adam optimizer that computes parameter updates using backpropagation.

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
        # Dictionaries to store the first (m) and second (v) moment estimates.
        self.moment1: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}

    def forward(self, model: nn.Module, loss_fn: Callable[[nn.Module], torch.Tensor]) -> Dict[str, torch.Tensor]:
        loss = loss_fn(model)
        param_list = [param for _, param in model.named_parameters()]
        grads = torch.autograd.grad(loss, param_list, create_graph=False)
        self.step += 1
        new_params: Dict[str, torch.Tensor] = {}

        for (name, param), grad in zip(model.named_parameters(), grads):
            if name not in self.moment1:
                self.moment1[name] = torch.zeros_like(param)
                self.v[name] = torch.zeros_like(param)
            moment1 = self.moment1[name]
            moment2 = self.moment2[name]
            new_moment1 = self.beta1 * moment1 + (1 - self.beta1) * grad
            new_moment2 = self.beta2 * moment2 + (1 - self.beta2) * (grad * grad)
            bias_corrected_moment1 = new_moment1 / (1 - self.beta1 ** self.step)
            bias_corrected_moment2 = new_moment2 / (1 - self.beta2 ** self.step)
            new_param = param - self.step_size * bias_corrected_moment1 / (torch.sqrt(bias_corrected_moment2) + self.eps)
            new_params[name] = new_param
            self.moment1[name] = new_moment1
            self.v[name] = new_moment2

        return new_params

if __name__ == "__main__":
    # Create an instance of the dummy model.
    model = DummyModel()

    # Create an instance of the Adam optimizer.
    adam_optimizer = AdamOpt(step_size=0.1)

    # Script the optimizer. Here we provide example inputs via a tuple:
    # (model, loss_fn_placeholder)
    scripted_adam = torch.jit.script(adam_optimizer)

    # Optionally, generate and save the computation graph.
    print(scripted_adam.graph)
    torch.jit.save(scripted_adam, __file__.replace('.py', '.pt'))
