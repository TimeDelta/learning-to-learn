from typing import List

import numpy as np
from fbm import fbm


def generate_complex_regression_data(n_samples=1000, latent_dim=5, observed_dim=3):
    """
    Generate random data with a nonlinear mapping from the latent space to the observed space.
    Returns:
         x: Input tensor of shape (n_samples, observed_dim)
         y: Target tensor of shape (n_samples, 1)
    """
    min_val = -np.pi / 2
    max_val = np.pi / 2
    latent_variables = np.random.uniform(low=min_val, high=max_val, size=(n_samples, latent_dim))

    noise_std = (max_val - min_val) / 10
    random_input_projection = np.random.randn(latent_dim, observed_dim)
    random_output_projection = np.random.randn(latent_dim, 1)

    nonlinear_input = np.sin(np.dot(latent_variables, random_input_projection))
    nonlinear_output = np.tanh(np.dot(latent_variables, random_output_projection))

    noisy_nonlinear_input = nonlinear_input + np.random.normal(scale=noise_std, size=nonlinear_input.shape)
    noisy_nonlinear_output = nonlinear_output + np.random.normal(scale=noise_std, size=nonlinear_output.shape)

    return noisy_nonlinear_input, noisy_nonlinear_output


def generate_fbm_sequence(
    means: List[float], stdevs: List[float], hurst_target: float, fbm_length: float, num_features: int, num_states: int
):
    assert len(means) == num_features
    assert len(stdevs) == num_features
    series = []
    for f in range(num_features):
        # fbm() returns array of length n+1: remove first value so series doesn't start at 0
        FBM = fbm(n=num_states, hurst=hurst_target, length=fbm_length, method="daviesharte")[1:]
        # scale and shift the series by stdev and mean for feature f
        series.append(means[f] + stdevs[f] * np.sin(FBM))
    return np.stack(series, axis=-1).astype(np.float32)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    assert generate_fbm_sequence(
        means=[2.3, 5], stdevs=[2, 14], hurst_target=0.3, fbm_length=np.pi, num_features=2, num_states=100
    ).shape == (100, 2)

    observed_dim = 2
    for l_dim in range(1, 5):
        X, y = generate_complex_regression_data(n_samples=1000, latent_dim=l_dim, observed_dim=observed_dim)
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  # color map has to be between 0 and 1
        if observed_dim == 2:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", alpha=0.7)
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        elif observed_dim == 3:
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="viridis", alpha=0.7)
            ax.set_xlabel("Feature 1")
            ax.set_ylabel("Feature 2")
            ax.set_zlabel("Feature 3")
        plt.colorbar(scatter, label="Target (y)")
        plt.show()
