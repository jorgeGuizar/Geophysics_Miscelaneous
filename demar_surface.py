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
file_name = "UTD_Demar_Mulach.txt"
data = np.loadtxt(file_name, delimiter=',')


data = data[:, ~np.isnan(data).any(axis=0)]
k=30000
print(data.shape)
X=data[:k,0]
Y=data[:k,1]
Z=data[:k,2]

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


#plt.imshow(zi, vmin=vmin, vmax=vmax)
#plt.colorbar(shrink=0.75)  # shrink makes the colourbar a bit shorter
#plt.show()


#methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

# Fixing random state for reproducibility
#np.random.seed(19680801)

#grid = np.random.rand(4, 4)

#fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
#                        subplot_kw={'xticks': [], 'yticks': []})

#for ax, interp_method in zip(axs.flat, methods):
#    ax.imshow(zi, interpolation=interp_method, cmap='viridis')
#    ax.set_title(str(interp_method))

#plt.tight_layout()
#plt.show()


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
 # FILTERING KERNELS


kernel = np.ones((3,3)) / 9
kernel25 = np.ones((3,3)) / 25


laplacian = np.array([[0, -1,  0],
                         [-1,  4,  -1],
                         [0, -1,  0]])

robinson = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]])

kirsch = np.array([[ 3,   3,  -1],
                      [ 3,   0,  -1],
                      [-1,  -1,  -1]])

sobel = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


median = scipy.signal.medfilt2d(zi,11)



gaussian = scipy.ndimage.gaussian_filter(zi, 3.0)

fft_output = scipy.signal.fftconvolve(zi, kernel)
new_output = scipy.signal.convolve2d(zi, kernel)
new_output_25 = scipy.signal.convolve2d(zi, kernel25)
shaded_relief = scipy.signal.convolve2d(zi, laplacian)
robinson_relief = scipy.signal.convolve2d(zi, robinson)
kirsch_element = scipy.signal.convolve2d(zi, kirsch)
sobel_relief = scipy.signal.convolve2d(zi, sobel)
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
grid = pv.StructuredGrid(xi, yi, zi)
grid.plot()


a=np.min(Z)
b=np.max(Z)
# Plot mean curvature as well
grid.plot_curvature(clim=[a, b])





