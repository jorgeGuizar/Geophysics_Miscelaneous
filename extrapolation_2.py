import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd 
import random
from scipy.interpolate import griddata
from datetime import datetime
from scipy.interpolate import CloughTocher2DInterpolator
from mpl_toolkits.mplot3d import axes3d
import scipy.ndimage
import scipy.signal
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton
import pyvista as pv
#from pyvista import examples
from scipy.interpolate import Rbf



def plot_3d(x, y, z, w=None, show=True):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure(figsize=(10, 6))
    ax = axes3d.Axes3D(fig)
    ax.scatter3D(x, y, z, c=w if not w is None else "b")
    plt.show()

print("Importing data...")
# Import data from file
file_name = "UTD_Demar_Mulach-A-ICA.txt"
data = np.loadtxt(file_name, delimiter=' ')
print(data.shape)

data = data[:, ~np.isnan(data).any(axis=0)]


X=data[:30000,0]
Y=data[:30000,1]
Z=data[:30000,2]*-1



# Definir los límites de la cuadrícula en X e Y
grid_x, grid_y = np.mgrid[min(X):max(X):500j, min(Y):max(Y):500j]
#grid_x, grid_y = np.meshgrid(X, Y)

# Interpolación de los valores Z en la cuadrícula
grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='cubic')



# Usar Rbf para la extrapolación (interpolación radial)
rbf_interpolator = Rbf(X, Y, Z, function='linear')

# Crear una cuadrícula extendida para la extrapolación
extended_grid_x, extended_grid_y = np.mgrid[
    (min(X)-100):(max(X)+100):500j,
    (min(Y)-100):(max(Y)+100):500j
]
extended_grid_z = rbf_interpolator(extended_grid_x, extended_grid_y)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(extended_grid_x, extended_grid_y, extended_grid_z, cmap='viridis')

plt.show()









#xp=[463570.3, 463665.8,463630.6,463690.2]
#yp=[2043676.9, 2043666.9,2044153.6,2044145.6]

#xi = np.linspace(min(xp), max(xp), 1000)
#yi = np.linspace(min(yp), max(yp), 1000)
#xi = np.linspace(min(X), max(X), 4000)
#yi = np.linspace(min(Y), max(Y), 1000)
#xi, yi = np.meshgrid(xi, yi)



#zi = griddata((X, Y), Z, (xi, yi), method='cubic')
#plot_3d(X, Y, Z)

#xii = xi.flatten()
#yii = yi.flatten()


#rbf3 = Rbf(X, Y, Z, function="multiquadric", smooth=5)
#znew = rbf3(xii, yii)

#zz=znew.reshape(len(xi),len(yi))

#print(znew.shape,xi.shape,yi.shape,zz.shape)

#plot_3d(xnew, ynew, znew)
# Step 3: Interpolate the Z values onto the grid
#zi = griddata((X, Y), Z, (xi, yi), method='cubic')


#rbf3 = Rbf(X, Y, Z, function="multiquadric", smooth=5)
#znew = rbf3(xi, yi)

#grid = pv.StructuredGrid(xi, yi, zz)
#grid.plot()
#fig=plt.figure(figsize=(12,7), dpi=100)
#ax = plt.axes(projection='3d')
 
#surf = ax.plot_surface(xi, xi,zz, cmap = 'jet', rstride=1, cstride=1, alpha=None,\
#                       linewidth=0, antialiased=True)
 
# Set axes label
#ax.set_title('Python Surface Plot (Syeilendra ITB)')
#ax.set_xlabel('x', labelpad=5)
#ax.set_ylabel('y', labelpad=5)
#ax.set_zlabel('z', labelpad=5)
 
#fig.colorbar(surf, shrink=0.7, aspect=15)
#plt.show()