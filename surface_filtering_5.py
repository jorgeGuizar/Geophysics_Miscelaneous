import time
import verde as vd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dask
import pyproj
import pooch
import random
from scipy.interpolate import griddata
import numpy as np
from numpy import linalg

###########################################################################################################################################################
############################################################################################################################################################
# FOLDER OF THE FILE
start=time.time()
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
###########################################################################################################################################################
############################################################################################################################################################
# FUNCTIONS 
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

def LS_solution_line_2(Z):    
    """_summary_
    
    Least squares solution for a linear model Z=m0+m1*X
    
    Args:
        X (_type_): _description_
        Y (_type_): _description_
        Z (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    X=np.arange(1,len(Z)+1,1)
        
    GTG=np.zeros((2,2));
    GTD=np.zeros(2);
    GTG[0,0]=len(X)
    GTG[0,1]=np.sum(X)
    GTG[1,0]=np.sum(X)
    GTG[1,1]=np.sum(X**2)
    GTD[0]=np.sum(Z)
    GTD[1]=np.sum(X*Z)
    I=np.eye(2)
    beta=.001
    IB=I*beta
    invGTG=np.linalg.inv(GTG+IB)
    m=np.dot(invGTG,GTD)
    
    return m,X



def LS_solution_PLANE(X,Y,Z):    
    """_summary_
    
    Least squares solution for a linear model a plane Z=m0+m1*X+m2*Y
    
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


###########################################################################################################################################################
############################################################################################################################################################
# read values of the dataframe
df=pd.read_csv(file_name,delimiter=',')
df = df.sort_values('Ping_Number')
###########################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
# CLEANING THE DATAFRAME
# remove the rows with NaN values       
df = df.dropna()

# remove the rows with 0 values
df = df[(df != 0).all(1)]   

##trend removal
Z_initial=df['Footprint_Z'].values
X=df['Footprint_X'].values
Y=df['Footprint_Y'].values

#plt.figure()
#plt.scatter(X, Y, c=Z_initial, s=2)
#plt.colorbar()
#plt.show()


#projection = pyproj.Proj(proj="merc")#, lat_ts=data.LAT.mean())
coordinates =(X,Y)
trend = vd.Trend(degree=1)
trend.fit(coordinates, Z_initial)
regional = trend.predict(coordinates)


#plt.figure()
#plt.scatter(X, Y, c=regional, s=2)
#plt.imshow(regional)
#plt.colorbar()
#plt.show()


# calculate the residual from the trend
residual =Z_initial - regional
scale = vd.maxabs(residual)


plt.figure()
#plt.scatter(X, Y, c=residual, s=2)
plt.scatter(coordinates[0], coordinates[1], c=residual, s=2, cmap="RdBu_r", vmin=-scale, vmax=scale)
#plt.imshow(regional)
plt.colorbar()
plt.show()



# block reductions or decimating the blocks

reducer = vd.BlockReduce("median", spacing=.5)
block_coords, block_bathymetry = reducer.filter(coordinates, Z_initial)


plt.figure()
plt.scatter(block_coords[0], block_coords[1], c=block_bathymetry, s=2)
plt.axis("scaled")
plt.colorbar()
plt.show()


# interpolate the data
spline = vd.Spline()

spline.fit(block_coords, block_bathymetry)
predicted = spline.predict(coordinates)

plt.figure()
plt.scatter(coordinates[0], coordinates[1], c=predicted, s=2)
plt.axis("scaled")
plt.colorbar()



end = time.time()
elapsed_time=end - start

print('Execution time:', elapsed_time/60, 'minutes')
