# Import a file with data in x,y,z and place it into a matrix to perform 2D fourier transform

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

print("Importing data...")
# Import data from file
file_name = "DTM_IchA_0.5m.xyz"
data = np.loadtxt(file_name, delimiter=',')

# Create a matrix with the data
x = data[:,0]
y = data[:,1]
z = data[:,2]

#  Find the index of X where X >= 557200 and X <= 557400
#  Find the index of Y where Y >= 2104400 and Y <= 2104600
min_x1 = 557200
max_x1 = 557400
min_y1 = 2104800
max_y1 = 2105000

print("Indexing data...")
index_x = np.where((x >= min_x1) & (x <= max_x1))
index_y = np.where((y >= min_x1) & (y <= max_y1))
index_xy = np.intersect1d(index_x, index_y)

x1=x[index_xy]
y1=y[index_xy]
z1=z[index_xy]

len_x1 = len(x1[x1==x1[0]])
len_y1 = len(y1[y1==y1[0]])

# Create a meshgrid
X1, Y1 = np.meshgrid(np.linspace(min(x1), max(x1), len_x1), np.linspace(min(y1), max(y1), len_y1))

# Interpolate Z values based on scattered data
Z1 = np.zeros_like(X1)

# You can use any interpolation method you prefer here, for example, linear interpolation
from scipy.interpolate import griddata
Z1 = griddata((x1, y1), z1, (X1, Y1), method='linear')

print("Saving data...")
# SaveX, Y, Z to file
np.savetxt('X1.txt', X1, delimiter=',')
np.savetxt('Y1.txt', Y1, delimiter=',')
np.savetxt('Z1.txt', Z1, delimiter=',')


print("Plotting data...")
# Plot the surface
fig = plt.figure()
ax = fig.gca()
ax.pcolormesh(X1, Y1, Z1, cmap='viridis', vmin=min(z1), vmax=max(z1))
plt.show()
