import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the forward gravity model
def forward_model(density_model, depths, gravitational_constant=6.67430e-11):
    """
    Computes gravity anomaly from a simple density model.

    Parameters:
        density_model (np.array): Subsurface density distribution (kg/m^3).
        depths (np.array): Depths of the density layers (m).
        gravitational_constant (float): Gravitational constant (default: 6.67430e-11).

    Returns:
        np.array: Gravity anomalies at the surface.
    """
    return gravitational_constant * np.sum(density_model / depths**2)

# Generate synthetic data
def generate_synthetic_data(true_density_model, depths, noise_std=1e-6):
    """
    Generate synthetic gravity anomaly data for a given density model.

    Parameters:
        true_density_model (np.array): True subsurface density distribution.
        depths (np.array): Depths of the density layers.
        noise_std (float): Standard deviation of noise.

    Returns:
        np.array: Observed gravity anomalies.
    """
    true_anomaly = forward_model(true_density_model, depths)
    noise = np.random.normal(0, noise_std, len(depths))
    return true_anomaly + noise

# Bayesian Inversion
def bayesian_inversion(observed_data, depths, prior_mean, prior_covariance, noise_std):
    """
    Perform Bayesian inversion to estimate the density model.

    Parameters:
        observed_data (np.array): Observed gravity anomaly data.
        depths (np.array): Depths of the density layers.
        prior_mean (np.array): Mean of the prior distribution for the density model.
        prior_covariance (np.array): Covariance of the prior distribution.
        noise_std (float): Standard deviation of the noise in observations.

    Returns:
        np.array, np.array: Posterior mean and covariance.
    """
    # Construct the sensitivity matrix (Jacobian)
    sensitivity_matrix = -2 * (1/depths**3)

    # Compute the noise covariance matrix
    noise_covariance = np.eye(len(observed_data)) * noise_std**2

    # Posterior covariance
    posterior_covariance = np.linalg.inv(
        np.linalg.inv(prior_covariance) + 
        sensitivity_matrix.T @ np.linalg.inv(noise_covariance) @ sensitivity_matrix
    )

    # Posterior mean
    posterior_mean = posterior_covariance @ (
        np.linalg.inv(prior_covariance) @ prior_mean +
        sensitivity_matrix.T @ np.linalg.inv(noise_covariance) @ observed_data
    )

    return posterior_mean, posterior_covariance

# Example Usage
if __name__ == "__main__":
    # True model parameters
    true_density_model = np.array([2000, 2200, 2500])  # Density in kg/m^3
    depths = np.array([500, 1000, 1500])  # Depths in meters

    # Generate synthetic data
    observed_data = generate_synthetic_data(true_density_model, depths)

    # Define prior distribution
    prior_mean = np.array([1800, 2000, 2300])
    prior_covariance = np.eye(len(prior_mean)) * 100**2

    # Perform Bayesian inversion
    posterior_mean, posterior_covariance = bayesian_inversion(
        observed_data, depths, prior_mean, prior_covariance, noise_std=1e-6
    )

    # Output results
    print("Observed Data:", observed_data)
    print("Posterior Mean:", posterior_mean)
    print("Posterior Covariance:", posterior_covariance)

    # Plot results
    plt.errorbar(range(len(posterior_mean)), posterior_mean, 
                 yerr=np.sqrt(np.diag(posterior_covariance)), fmt='o', label='Posterior')
    plt.scatter(range(len(true_density_model)), true_density_model, color='red', label='True Model')
    plt.legend()
    plt.xlabel('Layer Index')
    plt.ylabel('Density (kg/m^3)')
    plt.title('Bayesian Inversion Results')
    plt.show()
