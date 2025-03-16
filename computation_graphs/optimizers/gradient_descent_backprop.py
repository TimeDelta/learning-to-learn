import torch
import torch.nn as nn

class BackpropGD(nn.Module):
    def __init__(self, step_size=0.1):
        """
        Backprop-Based Gradient Descent on a per-parameter basis.
        - step_size: learning rate multiplier.
        """
        super(BackpropGD, self).__init__()
        self.step_size = step_size

    def forward(self, model, loss_fn):
        """
        compute new parameters
        """
        # create_graph=False for efficiency
        grads = torch.autograd.grad(loss_fn(model), model.parameters(), create_graph=False)

        new_params = {}
        for (name, param), grad in zip(model.named_parameters(), grads):
            new_params[name] = param - self.step_size * grad

        return new_params

loss_fn_placeholder = lambda model: (sum(param.sum() for param in model.parameters()) - 42)**2

if __name__ == "__main__":
    optimizer_module = BackpropGD()
    # generate dynamic computation graph:
    traced_module = torch.jit.script(optimizer_module, (torch.randn(5), loss_fn_placeholder))
    print(traced_module.graph)
    torch.jit.save(traced_module, __file__.replace('.py', '.pt'))
