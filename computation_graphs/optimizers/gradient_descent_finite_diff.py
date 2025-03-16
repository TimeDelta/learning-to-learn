import torch
import torch.nn as nn

class FiniteDifferenceGD(nn.Module):
    def __init__(self, epsilon=1e-4, step_size=0.1):
        """
        Finite Difference Gradient Descent:
        - epsilon: small perturbation used to compute finite differences.
        - step_size: learning rate multiplier.
        """
        super(FiniteDifferenceGD, self).__init__()
        self.epsilon = epsilon
        self.step_size = step_size

    def forward(self, model, loss_fn):
        """
        compute new params
        """
        for i in range(len(params)):
            loss_plus = loss_fn(params[i] + self.epsilon)
            loss_minus = loss_fn(params[i] - self.epsilon)
            # Finite-difference estimate of gradient:
            grad_estimate = (loss_plus - loss_minus) / (2 * self.epsilon)
            # Update: note the minus sign for gradient descent.
            update = -self.step_size * grad_estimate
            params[i] += update
        return params

if __name__ == "__main__":
    optimizer_module = FiniteDifferenceGD()
    # generate dynamic computation graph:
    traced_module = torch.jit.script(optimizer_module, (torch.randn(5), lambda model: 1))
    print(traced_module.graph)
    torch.jit.save(traced_module, __file__.replace('.py', '.pt'))
