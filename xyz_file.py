import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Archivo XYZ con 1,000,000 elementos (ejemplo)
path_xyz = r'C:\Users\LEGA\Documents\Geofisica\MB\CURVAEXPANSION.xyz'

datos = np.loadtxt(path_xyz,delimiter=',')
indices = np.random.choice(datos.shape[0], size=100000, replace=False)
datos_muestreados = datos[indices]

# Extraer coordenadas x, y, z
x = datos_muestreados[:, 0]
y = datos_muestreados[:, 1]
z = datos_muestreados[:, 2]

print("almost there")






# Encontrar el número de puntos únicos en x e y
#nx = len(np.unique(x))
#ny = len(np.unique(y))

# # Crear una cuadrícula 2D para z
# Z = z.reshape(nx, ny)

# # Graficar la superficie usando imshow
# plt.figure(figsize=(10, 8))
# plt.imshow(Z, extent=(np.min(x), np.max(x), np.min(y), np.max(y)), cmap='viridis', origin='lower')
# plt.colorbar(label='Z')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Superficie desde archivo XYZ')

# plt.show()
