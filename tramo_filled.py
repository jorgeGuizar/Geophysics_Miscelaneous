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

import pyvista as pv
#from pyvista import examples

####################################################################################################

print("Importing data...")
# Import data from file
file_name = "tramo_recortado.txt"
data = np.loadtxt(file_name, delimiter=' ')


data = data[:, ~np.isnan(data).any(axis=0)]
k=30000
print(data.shape)
X=data[:k,0]
Y=data[:k,1]
Z=data[:k,2]


X=data[:,0]
Y=data[:,1]
Z=data[:,2]*-1.0


# Step 1: Filter the data
# Filter the data to remove any NaN values
# crate a surface from x y z
# Create a meshgrid
# Interpolate Z values based on scattered data
# You can use any interpolation method you prefer here, for example, linear interpolation
# SaveX, Y, Z to file
# Plot the surfac
#yi=np.unique(Y)
xi = np.linspace(min(X), max(X), 1000)
yi = np.linspace(min(Y), max(Y), 1000)
xi, yi = np.meshgrid(xi, yi)

# Step 3: Interpolate the Z values onto the grid
zi = griddata((X, Y), Z, (xi, yi), method='cubic')


vmin = np.min(Z)
vmax = np.max(Z)
print ('min {0}, max {1}'.format(vmin, vmax))

median = scipy.signal.medfilt2d(zi,11)
grid = pv.StructuredGrid(xi, yi, zi)
grid.plot()
