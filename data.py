import numpy as np
import matplotlib.pyplot as plt

def generate_complex_loss_landscape_data(n_samples=1000, latent_dim=5, observed_dim=3):
    """
    Generate random data with a nonlinear mapping from the latent space to the observed space.
    Returns:
         x: Input tensor of shape (n_samples, observed_dim)
         y: Target tensor of shape (n_samples, 1)
    """
    min_val = -np.pi/2
    max_val = np.pi/2
    latent_variables = np.random.uniform(low=min_val, high=max_val, size=(n_samples, latent_dim))

    noise_std = (max_val-min_val) / 10
    if latent_dim < observed_dim:
        random_input_projection = np.random.randn(latent_dim, observed_dim)
        random_output_projection = np.random.randn(latent_dim, 1)
    else:
        random_input_projection = np.random.randn(latent_dim, latent_dim)
        random_output_projection = np.random.randn(latent_dim, 1)

    nonlinear_input = np.sin(np.dot(latent_variables, random_input_projection))
    nonlinear_output = np.tanh(np.dot(latent_variables, random_output_projection))

    noisy_nonlinear_input = nonlinear_input + np.random.normal(scale=noise_std, size=nonlinear_input.shape)
    noisy_nonlinear_output = nonlinear_output + np.random.normal(scale=noise_std, size=nonlinear_output.shape)

    if latent_dim < observed_dim: # create extra features using additional noise
        extra_feature_shape = (n_samples, observed_dim - latent_dim)
        extra_features_input = np.random.normal(scale=noise_std, size=extra_feature_shape)
        noisy_nonlinear_input = np.hstack((noisy_nonlinear_input, extra_features_input))

    return noisy_nonlinear_input, noisy_nonlinear_output

if __name__ == '__main__':
    observed_dim = 2
    for l_dim in range(1, 5):
        X, y = generate_complex_loss_landscape_data(n_samples=1000, latent_dim=l_dim, observed_dim=observed_dim)
        y = (y-np.min(y))/(np.max(y)-np.min(y)) # color map has to be between 0 and 1
        if observed_dim == 2:
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
        elif observed_dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis', alpha=0.7)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.set_zlabel('Feature 3')
        plt.colorbar(scatter, label='Target (y)')
        plt.show()
