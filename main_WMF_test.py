# Import a file with data in x,y,z and place it into a matrix to perform 2D fourier transform

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import import_data
import pandas as pd

COLORMAP = 'winter'
MINZ = -32.8
MAXZ = -31.6
MINZ_RESTA = -0.2
MAXZ_RESTA = 0.2

def plot_data(ax, X, Y, Z):
    # Plot the surface with colormap values
    # min of Z excluding NaNs
    # max of Z excluding NaNs
    Zmin1 = np.nanmin(Z)
    Zmax1 = np.nanmax(Z)

    ax.pcolormesh(X, Y, Z, cmap=COLORMAP, vmin=Zmin1, vmax=Zmax1, shading='auto')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def save_file(X, Y, Z):
    XX=list(X.flatten())
    YY=list(Y.flatten())
    ZZ=list(Z.flatten())

    # np.array(XX,YY,ZZ)
    CC=np.array([XX,YY,ZZ])
    CC=CC.T

    file =open('carmen.txt','w')
    for ci in CC:
        file.write(str(ci)+'\n')   
    file.close()

    #DF = pd.DataFrame(C)
    #DF.to_csv("data1.csv" delimiter)


    file =open('carmen.txt','w')
    for ci in CC:
        file.write(str(ci)+'\n')   
    file.close() 



print("Importing data...")
#X, Y, Z = import_data.import_data_small()
X, Y, Z = import_data.import_data_full()


print("Computing filters...")
WMF4 = ndimage.median_filter(Z, size=50, mode='constant', cval=0.0)
dif = Z - WMF4

# Select from dif only the values less than -0.1
# dif1 = np.where(dif > -0.05, dif, 0)
dif2 = np.where(dif > -0.10, dif, 0)
# dif3 = np.where(dif > -0.15, dif, 0)

resta = Z - dif2

print("Plotting data...")
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.colorbar(ax1.pcolormesh(X, Y, Z, cmap=COLORMAP, vmin=MINZ, vmax=MAXZ, shading='auto'), ax=ax1)
fig.colorbar(ax2.pcolormesh(X, Y, resta, cmap=COLORMAP, vmin=MINZ, vmax=MAXZ, shading='auto'), ax=ax2)

plt.show()

# Save X, Y, Z to a CSV file
save_file(X, Y, Z)




        