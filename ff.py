import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
nx, nz = 100, 100  # Grid size
dx, dz = 10.0, 10.0  # Grid spacing (m)
dt = 0.001  # Time step (s)
nt = 1000  # Number of time steps

# Physical parameters
rho = 2500.0  # Density (kg/m^3)
c_p = 3000.0  # P-wave velocity (m/s)
c_s = 1500.0  # S-wave velocity (m/s)
mu = rho * c_s ** 2  # Shear modulus (Pa)
lambda_ = rho * c_p ** 2 - 2 * mu  # First Lame parameter (Pa)
alpha = 0.02  # Damping coefficient for PML

# PML parameters
pml_width = 10  # Width of PML layer (grid points)

# Initialize wavefields
vx = np.zeros((nx + 2*pml_width, nz + 2*pml_width))
vz = np.zeros((nx + 2*pml_width, nz + 2*pml_width))
sigma_xx = np.zeros((nx + 2*pml_width, nz + 2*pml_width))
sigma_zz = np.zeros((nx + 2*pml_width, nz + 2*pml_width))
sigma_xz = np.zeros((nx + 2*pml_width, nz + 2*pml_width))

# Initialize PML damping profiles
pml_damping_x = np.ones(nx + 2*pml_width)
pml_damping_z = np.ones(nz + 2*pml_width)

for i in range(pml_width):
    damping_value = alpha * ((pml_width - i) / pml_width) ** 2
    pml_damping_x[i] = damping_value
    pml_damping_x[-(i+1)] = damping_value
    pml_damping_z[i] = damping_value
    pml_damping_z[-(i+1)] = damping_value

# Source parameters
src_x, src_z = nx // 2, nz // 2
src_x += pml_width
src_z += pml_width
source_time_function = np.exp(-((np.arange(nt) - 50) / 10) ** 2)  # Gaussian pulse

# Finite-difference loop
for it in range(nt):
    # Apply source
    vx[src_x, src_z] += source_time_function[it] / (rho * dx)
    
    # Update velocity fields using finite-difference approximations
    vx[1:-1, 1:-1] += dt / rho * (sigma_xx[1:-1, 1:-1] - sigma_xx[:-2, 1:-1]) / dx
    vz[1:-1, 1:-1] += dt / rho * (sigma_zz[1:-1, 1:-1] - sigma_zz[1:-1, :-2]) / dz
    
    # Update stress fields using finite-difference approximations
    sigma_xx[1:-1, 1:-1] += lambda_ * (vx[1:-1, 1:-1] - vx[:-2, 1:-1]) / dx + mu * (vx[1:-1, 1:-1] - vx[:-2, 1:-1]) / dx
    sigma_zz[1:-1, 1:-1] += lambda_ * (vz[1:-1, 1:-1] - vz[1:-1, :-2]) / dz + mu * (vz[1:-1, 1:-1] - vz[1:-1, :-2]) / dz
    sigma_xz[1:-1, 1:-1] += mu * ((vx[1:-1, 2:] - vx[1:-1, :-2]) / (2 * dz) + (vz[2:, 1:-1] - vz[:-2, 1:-1]) / (2 * dx))
    
    # Apply PML to velocity fields
    vx *= pml_damping_x[:, np.newaxis]
    vz *= pml_damping_z[np.newaxis, :]
    
    # Apply PML to stress fields
    sigma_xx *= pml_damping_x[:, np.newaxis]
    sigma_zz *= pml_damping_z[np.newaxis, :]
    sigma_xz *= np.sqrt(pml_damping_x[:, np.newaxis] * pml_damping_z[np.newaxis, :])

    # Visualization (optional)
    if it % 100 == 0:
        plt.imshow(vx.T, cmap='seismic', aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'Time step {it}')
        plt.show()

# Final output and analysis
