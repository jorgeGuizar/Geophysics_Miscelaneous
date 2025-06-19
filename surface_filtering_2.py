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
from numpy import linalg
import skfda



print("Importing data...")
# Import data from file
file_name = r'C:\Users\LEGA\Documents\Geofisica\MB\etkal\0077 - L11B-SAAKUN - 0001_752.txt'

#data = np.loadtxt(file_name, delimiter=',')


#data = data[:, ~np.isnan(data).any(axis=0)]
#k=30000
#print(data.shape)

#X=data[:,1]
#Y=data[:,2]
#Z=data[:,3]


def LS_solution_line(X,Z):    
    """_summary_
    
    Least squares solution for a linear model Z=m0+m1*X
    
    Args:
        X (_type_): _description_
        Y (_type_): _description_
        Z (_type_): _description_

    Returns:
        _type_: _description_
    """
    GTG=np.zeros((2,2));
    GTD=np.zeros(2);
    GTG[0,0]=len(X)
    GTG[0,1]=np.sum(X)
    GTG[1,0]=np.sum(X)
    GTG[1,1]=np.sum(X**2)
    GTD[0]=np.sum(Z)
    GTD[1]=np.sum(X*Z)
    invGTG=np.linalg.inv(GTG)
    m=np.dot(invGTG,GTD)
    return m



def LS_solution_PLANE(X,Y,Z):    
    """_summary_
    
    Least squares solution for a linear model Z=m0+m1*X
    
    Args:
        X (_type_): _description_
        Y (_type_): _description_
        Z (_type_): _description_

    Returns:
        _type_: _description_
    """
    GTG=np.zeros((3,3));
    GTD=np.zeros(3);
    
    GTG[0,0]=len(X)
    GTG[0,1]=np.sum(X)
    GTG[0,2]=np.sum(Y)
    
    GTG[1,0]=np.sum(X)
    GTG[1,1]=np.sum(X**2)
    GTG[1,2]=np.sum(X*Y)

    GTG[2,0]=np.sum(Y)
    GTG[2,1]=np.sum(X*Y)
    GTG[2,2]=np.sum(X**2)
    
    
    GTD[0]=np.sum(Z)
    GTD[1]=np.sum(X*Z)
    GTD[2]=np.sum(Y*Z)
    invGTG=np.linalg.inv(GTG)
    m=np.dot(invGTG,GTD)
    return m

    



df=pd.read_csv(file_name,delimiter=',')
df = df.sort_values('Ping_Number')
print(df)
print(df.Ping_Number.unique(),len(df.Ping_Number.unique()))
#pinger=[]
pinger=df.Ping_Number.unique()
print(pinger,type(pinger))
z_inv=[]
for ping in pinger:
    df_filter=df[df.Ping_Number==ping]
    Y_p = df_filter['Footprint_Y'].values
    X_p=df_filter['Footprint_X'].values
    Z_p = df_filter['Footprint_Z'].values
    m=LS_solution_line(X_p,Z_p)    
    Zinv=m[0]+m[1]*X_p
    for z in Zinv:
        z_inv.append(z)
    #print (m)
    #print(Zinv)
    #pause=input("Press Enter to continue")
    #plt.plot(X_p,Z_p,'o')
    #plt.plot(X_p,Zinv,'-r')
    #plt.show()    
df=df.assign(Z_inv=z_inv)

Y_p = df['Footprint_Y'].values
X_p=df['Footprint_X'].values
Z_p =df['Z_inv'].values


#m_plane=LS_solution_line(Y_p,Z_p) 
m_plane=LS_solution_PLANE(X_p,Y_p,Z_p)
Z_inv_plane=m_plane[0]+(m_plane[1]*X_p)+(m_plane[2]*Y_p)
#Z_inv_plane=m_plane[0]+m_plane[1]*Y_p

df=df.assign(Z_inv_2=Z_inv_plane)


    
Y= df['Footprint_Y'].values
X=df['Footprint_X'].values
Z= df['Z_inv_2'].values

print(df.head())
    

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
print("Grdidding data .......")

vmin = np.min(Z)
vmax = np.max(Z)
print ('min {0}, max {1}'.format(vmin, vmax))
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
# FILTERING KERNELS


# kernel = np.ones((3,3)) / 9

# kernel25 = np.ones((3,3)) / 25


# laplacian = np.array([[0, -1,  0],
#                          [-1,  4,  -1],
#                          [0, -1,  0]])

# robinson = np.array([[-1, -1, -1],
#                         [ 0,  0,  0],
#                         [ 1,  1,  1]])

# kirsch = np.array([[ 3,   3,  -1],
#                       [ 3,   0,  -1],
#                       [-1,  -1,  -1]])

# sobel = np.array([[-1, 0, 1],
#                      [-2, 0, 2],
#                      [-1, 0, 1]])
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
# median = scipy.signal.medfilt2d(zi,11)
# gaussian = scipy.ndimage.gaussian_filter(zi, 3.0)
# fft_output = scipy.signal.fftconvolve(zi, kernel)
# new_output = scipy.signal.convolve2d(zi, kernel)
# new_output_25 = scipy.signal.convolve2d(zi, kernel25)
# shaded_relief = scipy.signal.convolve2d(zi, laplacian)
# robinson_relief = scipy.signal.convolve2d(zi, robinson)
# kirsch_element = scipy.signal.convolve2d(zi, kirsch)
# sobel_relief = scipy.signal.convolve2d(zi, sobel)




# def laplacian_smooth(vertices, iterations=5):
    
#     for _ in range(iterations):
#         new_vertices = np.zeros_like(vertices)
#         for i in range(1, vertices.shape[0] - 1):
#             for j in range(1, vertices.shape[1] - 1):
#                 new_vertices[i, j] = 0.25 * (vertices[i-1, j] + vertices[i+1, j] + 
#                                            vertices[i, j-1] + vertices[i, j+1])
#         vertices = new_vertices
#     return vertices


# laplacian = laplacian_smooth(zi, iterations=10)



# def moving_average_2d(data, window_size):
#     return np.convolve(data.ravel(), np.ones(window_size), mode='same').reshape(data.shape) / window_size


# moving_av = moving_average_2d(zi, 5)


# def smooth_exp(data, alpha=0.5):
#     YH= np.zeros_like(data)
#     YH[0]=data[0]
#     for i in range(1,len(data)):
#         YH[i]=alpha*data[i]+(1.0-alpha)*YH[i-1]
#     return YH


# #from scipy.signal import savgol_filter
# #zavgol = savgol_filter(zi, window_length=15, polyorder=3, axis=0)
# #zavgol = savgol_filter(zavgol, window_length=15, polyorder=3, axis=1)

# ft = np.fft.ifftshift(zi)
# ft = np.fft.fft2(ft)
# ft = np.fft.fftshift(ft)
# #####################################################################################################################
# #####################################################################################################################
# #####################################################################################################################

# plt.subplot(122)
# plt.imshow(abs(ft))
# #plt.xlim([480, 520])
# #plt.ylim([520, 480])  # Note, order is reversed for y
# plt.show()


plt.imshow(zi,vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title("LS solution")
plt.show()



# #plt.imshow(zavgol,vmin=vmin,vmax=vmax)
# #plt.title("zavgol")
# #plt.colorbar()
# #plt.show()


# plt.imshow(gaussian,vmin=vmin,vmax=vmax)
# plt.title("Gaussian")
# plt.colorbar()
# plt.show()


# plt.imshow(median,vmin=vmin,vmax=vmax)
# plt.title("Median")
# plt.colorbar()
# plt.show()


# plt.imshow(moving_av,vmin=vmin,vmax=vmax)
# plt.title("Moving average 2D")
# plt.colorbar()
# plt.show()



# plt.imshow(new_output,vmin=vmin,vmax=vmax)
# plt.title("Kernel")
# plt.colorbar()
# plt.show()



# plt.imshow(laplacian,vmin=vmin,vmax=vmax)
# plt.title("Laplacian")
# plt.colorbar()
# plt.show()



# #grid = pv.StructuredGrid(xi, yi,median)
# #grid.plot()


