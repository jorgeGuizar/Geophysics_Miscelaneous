import numpy as np
import matplotlib.pyplot as plt

#Parámetros de la simulación
nx = 200  # Número de puntos en la dirección x
nz = 200  # Número de puntos en la dirección z
dx = 10.0  # Espaciamiento en la dirección x (m)
dz = 10.0  # Espaciamiento en la dirección z (m)
dt = 0.001  # Paso de tiempo (s)
nt = 1000  # Número de pasos de tiempo

#Velocidades de onda P y S
vp = 3000.0  # Velocidad de onda P (m/s)
vs = 1500.0  # Velocidad de onda S (m/s)

#Densidad
rho = 2500.0  # Densidad (kg/m^3)

#Coeficientes de Lamé
mu = rho * vs**2
lamb = rho * vp**2 - 2 * mu

#Inicialización de los campos de velocidad y estrés
vx = np.zeros((nx, nz))
vz = np.zeros((nx, nz))
txx = np.zeros((nx, nz))
tzz = np.zeros((nx, nz))
txz = np.zeros((nx, nz))

#Fuente
x_src = nx // 2
z_src = nz // 2
f0 = 10.0  # Frecuencia central de la fuente (Hz)

#PML
npml = 20  # Número de puntos PML
R = 1e-6  # Reflectividad en la frontera PML

#Coeficientes PML
def pml_coeff(d, npml):
    coeff = np.zeros(npml)
    for i in range(npml):
        coeff[i] = (i / npml)**3 * np.log(R) / (2 * d)
    return coeff

fig, ax = plt.subplots()
plt.imshow(vx, cmap='RdBu', extent=(0, nx*dx, 0, nz*dz),animated=True)

#Simulación
for n in range(nt):
    # Fuente
    t = n * dt
    src = np.sin(2 * np.pi * f0 * t) * np.exp(-(t - 1 / f0)**2 / (2 * (1 / f0)**2))
    
    # Actualización de los campos de velocidad
    vx[x_src, z_src] += src * dt / rho
    vz[x_src, z_src] += src * dt / rho
    
    # Actualización de los campos de estrés
    txx[1:-1, 1:-1] += dt * (lamb * (vx[1:-1, 1:-1] - vx[:-2, 1:-1]) / dx + lamb * (vz[1:-1, 1:-1] - vz[1:-1, :-2]) / dz)
    tzz[1:-1, 1:-1] += dt * (lamb * (vx[1:-1, 1:-1] - vx[:-2, 1:-1]) / dx + lamb * (vz[1:-1, 1:-1] - vz[1:-1, :-2]) / dz)
    txz[1:-1, 1:-1] += dt * (mu * (vx[1:-1, 1:-1] - vx[1:-1, :-2]) / dz + mu * (vz[1:-1, 1:-1] - vz[:-2, 1:-1]) / dx)
    
    # Aplicación de las condiciones PML
    pml_x = pml_coeff(dx, npml)
    pml_z = pml_coeff(dz, npml)
    for i in range(npml):
        vx[i, :] *= np.exp(-pml_x[i] * dt)
        vx[-i-1, :] *= np.exp(-pml_x[i] * dt)
        vz[:, i] *= np.exp(-pml_z[i] * dt)
        vz[:, -i-1] *= np.exp(-pml_z[i] * dt)
        
    plt.imshow(vx, cmap='RdBu', extent=(0, nx*dx, 0, nz*dz))
plt.show()

#Visualización
plt.imshow(vx, cmap='RdBu', extent=(0, nx*dx, 0, nz*dz))
plt.colorbar()
plt.show()