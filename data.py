import numpy as np
import torch

def generate_oscillatory_data(n_samples: int = 200):
    """
    Returns:
         x: Input tensor of shape (n_samples, 1)
         y: Target tensor of shape (n_samples, 1)
    """
    x = np.linspace(-5, 5, n_samples)
    y = np.sin(2 * x) + 0.5 * np.sin(5 * x)
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    return x_tensor, y_tensor
