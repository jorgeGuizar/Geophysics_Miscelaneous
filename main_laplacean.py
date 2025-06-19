# Import a file with data in x,y,z and place it into a matrix to perform 2D fourier transform

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import import_data

COLORMAP = 'winter'
MINZ = -32.8
MAXZ = -31.6

def plot_data(ax, X, Y, Z):
    # Plot the surface with colormap values
    # min of Z excluding NaNs
    # max of Z excluding NaNs
    Zmin1 = np.nanmin(Z)
    Zmax1 = np.nanmax(Z)

    ax.pcolormesh(X, Y, Z, cmap=COLORMAP, vmin=Zmin1, vmax=Zmax1, shading='auto')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    

    # ax.imshow(Z, cmap='viridis')

print("Importing data...")
X, Y, Z = import_data.import_data_small()
#X, Y, Z = import_data.import_data_full()

# Save Z as jpg file    
# plt.imshow(Z, cmap='viridis')
# plt.show()
# plt.savefig('Z.jpg')



from scipy import ndimage, datasets
import matplotlib.pyplot as plt

result = ndimage.laplace(Z)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# Plot the surface
print("Plotting data...")
# plot_data(ax1, X, Y, Z)
plot_data(ax1, X, Y, Z + Z)
plot_data(ax2, X, Y, Z + result)

fig.colorbar(ax1.pcolormesh(X, Y, Z, cmap=COLORMAP, vmin=MINZ, vmax=MAXZ, shading='auto'), ax=ax1)
fig.colorbar(ax2.pcolormesh(X, Y, result, cmap='PRGn', vmin=-0.5, vmax=0.5, shading='auto'), ax=ax2)

plt.show()