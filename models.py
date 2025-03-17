import torch
import torch.nn as nn
import numpy as np

class ManyLossMinimaModel(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 20, output_dim: int = 1):
        """
        A simple feed-forward model with a sine activation in the first layer to create periodicity,
        followed by a tanh activation. The resulting output is a linear combination of these nonlinear
        features.
        """
        super(ManyLossMinimaModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Using sine activation to induce oscillatory behavior (and hence many local minima).
        # This will allow certain optimization improvements like momentum to have an advantage.
        x = torch.sin(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x
