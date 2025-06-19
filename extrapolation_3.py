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
from scipy.interpolate import RegularGridInterpolator
import time
import skfda



# Save timestamp
start = time.time()
print("Importing data...")
# Import data from file
file_name = "UTD_Demar_Mulach-A-ICA.txt"
data = np.loadtxt(file_name, delimiter=' ')
print(data.shape)

data = data[:, ~np.isnan(data).any(axis=0)]


X=data[:,0]
Y=data[:,1]
Z=data[:,2]*-1


print(min(X),max(X),min(Y),max(Y),min(Z),max(Z))

#g, yg ,zg = np.meshgrid(X, Y, Z, indexing='ij', sparse=True)
grid_x, grid_y = np.mgrid[min(X)-20:max(X)+20:5000j, min(Y):max(Y):5000j]
#grid_x, grid_y = np.meshgrid(X, Y)

grid_z = griddata((X, Y), Z, (grid_x, grid_y),method='cubic')


plt.imshow(grid_z)#, extent=(min(X)-20, max(X)+20, min(Y), max(Y),min(Z),max(Z)), origin='lower')
plt.show()



#fd_surface = skfda.FDataGrid([grid_z],( X, Y))

#plt.imshow(grid_z, extent=(min(X), max(X), min(Y), max(Y),min(Z),max(Z)), origin='lower')
#plt.show()

#interp = RegularGridInterpolator((X, Y),grid_z , bounds_error=False, fill_value=None)
#grid_x, grid_y = np.mgrid[min(X)-10:max(X)+10:500j, min(Y)-10:max(Y)+10:500j]
#grid_x, grid_y = np.meshgrid(X, Y)
#interp = RegularGridInterpolator((X, Y, Z), grid_z)

#xp=[463570.3, 463665.8,463630.6,463690.2]
#yp=[2043676.9, 2043666.9,2044153.6,2044145.6]

#xi = np.linspace(min(xp), max(xp), 1000)
#yi = np.linspace(min(yp), max(yp), 1000)
#xi = np.linspace(min(X), max(X), 4000)
#yi = np.linspace(min(Y), max(Y), 1000)
#xi, yi = np.meshgrid(xi, yi)



#zi = griddata((X, Y), Z, (xi, yi), method='cubic')
#plot_3d(X, Y, Z)

xii = grid_x.flatten()
yii = grid_y.flatten()
zii=grid_z.flatten()
print(xii.shape,yii.shape,zii.shape)
# Creation of FDataGrid


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

#surf = ax.plot_surface(grid_x, grid_y,grid_z, cmap = 'jet', rstride=1, cstride=1, alpha=None,\
#                       linewidth=0, antialiased=True)

# Set axes label
#ax.set_title('Python Surface Plot (Syeilendra ITB)')
#ax.set_xlabel('x', labelpad=5)
#ax.set_ylabel('y', labelpad=5)
#ax.set_zlabel('z', labelpad=5)
 
#fig.colorbar(surf, shrink=0.7, aspect=15)
#plt.show()

# Combine the vectors into a single 2D array (as columns)
data = np.column_stack((xii, yii, zii))
# Specify the filename
filename = 'vectors_columns_mulach.txt'
# Write the vectors to a text file as columns
np.savetxt(filename, data, fmt='%.2f', comments='')

print(f"Vectors written to {filename} as columns")

# Save timestamp
end = time.time()

