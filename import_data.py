# Import a file with data in x,y,z and place it into a matrix to perform 2D fourier transform

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage

def import_data_full():
    # Import data from file
    file_name = "DTM_IchA_0.5m.xyz"
    file_name = "10-09-23_IchA_DTMc_0.5m.xyz"

    data = np.loadtxt(file_name, delimiter=',')

    # Create a matrix with the data
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    x1 = np.unique(x)
    y1 = np.unique(y)

    # Create a meshgrid with the actual different values of x and y
    len_x = len(np.unique(x1))
    len_y = len(np.unique(y1))

    # Create a meshgrid
    X, Y = np.meshgrid(x1, y1, sparse=True)

    # Create a matrix with the values of z
    # Interpolate the values of z in the meshgrid X,Y
    from scipy.interpolate import griddata
    Z = griddata((x, y), z, (X, Y), method='linear')
     
    return X, Y, Z

def import_data_small():
    file_name = "X1.txt"
    X = np.loadtxt(file_name, delimiter=',')

    file_name = "Y1.txt"
    Y = np.loadtxt(file_name, delimiter=',')

    file_name = "Z1.txt"
    Z = np.loadtxt(file_name, delimiter=',')

    return X, Y, Z

# If I run from this file, execute the following
# import_data_full  
if __name__=='__main__':
    X,Y,Z = import_data_full()
    print(X)
    print(Y)
    print(Z)
