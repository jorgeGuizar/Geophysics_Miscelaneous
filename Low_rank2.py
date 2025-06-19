import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from scipy.signal import hilbert
from skimage.util import view_as_windows

# Function to generate synthetic sub-bottom sonar data with noise
def generate_synthetic_data(signal_length=500, num_traces=100, noise_level=0.2):
    """
    Generate synthetic sub-bottom sonar data with noise.

    Parameters:
        signal_length (int): Number of samples per trace.
        num_traces (int): Number of traces.
        noise_level (float): Noise amplitude relative to signal.

    Returns:
        np.ndarray: Noisy sonar data matrix.
    """
    time = np.linspace(0, 1, signal_length)
    data = np.zeros((signal_length, num_traces))
    
    for i in range(num_traces):
        freq = np.random.uniform(5, 25)  # Random frequency
        amplitude = np.random.uniform(0.5, 1.0)  # Random amplitude
        phase = np.random.uniform(0, 2 * np.pi)  # Random phase
        signal = amplitude * np.sin(2 * np.pi * freq * time + phase)
        data[:, i] = signal

    noise = noise_level * np.random.randn(*data.shape)
    return data + noise

# Function to compute the envelope of sonar data
def compute_envelope(data):
    """
    Compute the envelope of sonar data using the Hilbert transform.

    Parameters:
        data (np.ndarray): Sonar data matrix (samples x traces).

    Returns:
        np.ndarray: Envelope of the sonar data.
    """
    analytic_signal = hilbert(data, axis=0)
    envelope = np.abs(analytic_signal)
    return envelope

# Non-local low-rank denoising for sonar images
def non_local_low_rank_denoise(data, patch_size, rank):
    """
    Denoise sonar image data using a non-local low-rank algorithm.

    Parameters:
        data (np.ndarray): Sonar data matrix (samples x traces).
        patch_size (tuple): Size of the patches (height, width).
        rank (int): Desired rank for low-rank approximation.

    Returns:
        np.ndarray: Denoised sonar data matrix.
    """
    # Extract patches
    patches = view_as_windows(data, patch_size, step=1)
    patches_shape = patches.shape
    patches_reshaped = patches.reshape(-1, patch_size[0] * patch_size[1])

    # Apply low-rank approximation to each patch
    denoised_patches = []
    for patch in patches_reshaped:
        U, S, Vt = svd(patch.reshape(-1, 1), full_matrices=False)
        S_reduced = np.zeros_like(S)
        S_reduced[:rank] = S[:rank]
        denoised_patch = (U @ np.diag(S_reduced) @ Vt).flatten()
        denoised_patches.append(denoised_patch)

    denoised_patches = np.array(denoised_patches).reshape(patches_shape[:-2] + patch_size)

    # Reconstruct the denoised image
    denoised_data = np.zeros_like(data)
    counts = np.zeros_like(data)
    for i in range(denoised_patches.shape[0]):
        for j in range(denoised_patches.shape[1]):
            denoised_data[i:i+patch_size[0], j:j+patch_size[1]] += denoised_patches[i, j]
            counts[i:i+patch_size[0], j:j+patch_size[1]] += 1

    denoised_data /= counts
    return denoised_data

# Example Usage
if __name__ == "__main__":
    # Generate synthetic sonar data
    num_samples = 500
    num_traces = 50
    noise_level = 0.3

    noisy_data = generate_synthetic_data(num_samples, num_traces, noise_level)

    # Compute the envelope of the sonar data
    envelope_data = compute_envelope(noisy_data)

    # Apply non-local low-rank denoising
    patch_size = (10, 10)
    desired_rank = 5
    denoised_data = non_local_low_rank_denoise(envelope_data, patch_size, desired_rank)

    # Plot results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(envelope_data, aspect='auto', cmap='seismic', extent=[0, num_traces, num_samples, 0])
    plt.title("Noisy Envelope Data")
    plt.xlabel("Trace")
    plt.ylabel("Sample")

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_data, aspect='auto', cmap='seismic', extent=[0, num_traces, num_samples, 0])
    plt.title("Denoised Envelope Data")
    plt.xlabel("Trace")
    plt.ylabel("Sample")

    plt.tight_layout()
    plt.show()
