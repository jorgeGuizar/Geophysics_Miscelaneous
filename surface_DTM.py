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

import skfda



print("Importing data...")
# Import data from file
file_name = r'C:\Users\LEGA\Documents\Geofisica\MB\etkal\0077 - L11B-SAAKUN - 0001_752.txt'

df=pd.read_csv(file_name,delimiter=',')
print(df.Ping_Number.unique(),len(df.Ping_Number.unique()))
df_filter=df[df.Ping_Number==1774]
Y_p = df_filter['Footprint_Y'].values
X_p=df_filter['Footprint_X'].values
Z_p = df_filter['Footprint_Z'].values
#grid_x, grid_y = np.mgrid[min(X_p):max(X_p):5000j, min(Y_p):max(Y_p):5000j]
X_p=np.unique(X_p)
Y_p=np.unique(Y_p)
grid_x, grid_y = np.meshgrid(X_p, Y_p)
Z=Z_p.reshape(len(X_p),len(Y_p))
Z=np.transpose(Z)
#grid_z = griddata((X_p, Y_p), Z_p, (grid_x, grid_y),method='cubic')

#plt.imshow(grid_z)#, extent=(min(X)-20, max(X)+20, min(Y), max(Y),min(Z),max(Z)), origin='lower')

#plt.show()


#data = np.loadtxt(file_name, delimiter=',', skiprows=1)
#print(data.shape)

#data = data[:, ~np.isnan(data).any(axis=0)]


#beam=data[:,0]
#X=data[:,1]
#Y=data[:,2]
#Z=data[:,3]
#ping=data[:,4]

#print(min(X),max(X),min(Y),max(Y),min(Z),max(Z))


#grid_x, grid_y = np.mgrid[min(X):max(X):2000j, min(Y):max(Y):2000j]
#grid_z = griddata((X, Y), Z, (grid_x, grid_y),method='cubic')