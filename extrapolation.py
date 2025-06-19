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


X=data[:1000,0]
Y=data[:1000,1]
Z=data[:1000,2]*-1.0

xp=[463570.3, 463665.8,463630.6,463690.2]
yp=[2043676.9, 2043666.9,2044153.6,2044145.6]

xi = np.linspace(min(xp), max(xp), 4000)
yi = np.linspace(min(yp), max(yp), 4000)
#xi = np.linspace(min(X), max(X), 4000)
#yi = np.linspace(min(Y), max(Y), 4000)
xi, yi = np.meshgrid(xi, yi)



#zi = griddata((X, Y), Z, (xi, yi), method='cubic')
#plot_3d(X, Y, Z)

xii = xi.flatten()
yii = yi.flatten()


rbf3 = Rbf(X, Y, Z, function="multiquadric", smooth=5)
znew = rbf3(xii, yii)

#plot_3d(xnew, ynew, znew)
# Step 3: Interpolate the Z values onto the grid
#zi = griddata((X, Y), Z, (xi, yi), method='cubic')


#rbf3 = Rbf(X, Y, Z, function="multiquadric", smooth=5)
#znew = rbf3(xi, yi)

#grid = pv.StructuredGrid(xi, yi, zi)
#grid.plot()


fig=plt.figure(figsize=(12,7), dpi=100)
ax = plt.axes(projection='3d')

surf = ax.plot_surface(xii, yii, znew, cmap = 'jet', rstride=1, cstride=1, alpha=None,\
                       linewidth=0, antialiased=True)

# Set axes label
ax.set_title('Python Surface Plot (Syeilendra ITB)')
ax.set_xlabel('x', labelpad=5)
ax.set_ylabel('y', labelpad=5)
ax.set_zlabel('z', labelpad=5)

fig.colorbar(surf, shrink=0.7, aspect=15)

plt.show()

