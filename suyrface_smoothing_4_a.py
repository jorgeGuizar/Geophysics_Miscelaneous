import time
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
from sklearn.linear_model import LinearRegression
from scipy.interpolate import RegularGridInterpolator
from numpy import linalg
import skfda
###########################################################################################################################################################
############################################################################################################################################################
# FOLDER OF THE FILE
start=time.time()
print("Importing data...")
# Import data from file
#file_name = r'C:\Users\LEGA\Documents\Geofisica\MB\etkal\0077 - L11B-SAAKUN - 0001_752.txt'
file_name = r'C:\Users\LEGA\Documents\Geofisica\MB\etkal\PRUEBA_evento.txt'
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
    while True:
        try:
            invGTG=np.linalg.inv(GTG)
            m=np.dot(invGTG,GTD)
            return m
        except np.linalg.LinAlgError:
            print("Singular matrix")
            invGTG=np.linalg.inv(GTG+(np.eye(2)*.001))
            m=np.dot(invGTG,GTD)
            return m
            #print(GTG)
    
    #m=np.dot(invGTG,GTD)
    #return m

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
df_out=pd.DataFrame(columns=["Beam","Ping_Number","Footprint_X","Footprint_Y","Footprint_Z","Z_reg"])
###########################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
# CLEANING THE DATAFRAME

print(df)
print(df.Ping_Number.unique(),len(df.Ping_Number.unique()))
#pinger=[]
pinger=df.Ping_Number.unique()
print(pinger,type(pinger))
z_inv=[]
prom_ping=[]
dif_depth=[]
for ping in pinger:
    df_filter=df[df.Ping_Number==ping]
    #Y_p = df_filter['Footprint_Y'].values
    X_p=df_filter['Footprint_X'].values
    Z_p = df_filter['Footprint_Z'].values
    beam=df_filter['Beam'].values
    mline=LS_solution_line(beam,Z_p)
    
    
    Z_inv_line=mline[0]+(mline[1]*beam)
    
    promedio_ping=np.mean(Z_inv_line)
    
    promedio_vec=np.ones(len(Z_inv_line))*promedio_ping
    
    Z_diff=np.abs(Z_inv_line-Z_p)
    
    #plt.plot(beam,Z_p,'b*',label='DATA')
    #plt.plot(beam,Z_inv_line,'r-',label='MODEL')
    #plt.show()
    
    for z,dif in zip (Z_inv_line,Z_diff):
        z_inv.append(z)
        #prom_ping.append(prom)
        dif_depth.append(dif)
        
# create a dataframe with the new values
# aasign the new values to the dataframe
# Z_inv: first solution of the plane
# Mean_ping: mean value of the plane
# Diff_depth: difference between the model and the plane
df=df.assign(Z_inv=z_inv)
#df=df.assign(Mean_ping=prom_ping)
df=df.assign(Diff_depth=dif_depth)
print(df.head())
# filter by difference in depth between the model and >.4 m are discarded
df=df[df['Diff_depth']<0.40] # initially it was .4 but these tests were run with .10
#3df=df[(df.Beam<800) & (df.Beam>200)]#df[(df.val < 0.5) | (df.val2 == 7)]


print(df.head())
print(np.min(dif_depth),np.max(dif_depth))
print(len)
df.drop(['Z_inv'],axis=1,inplace=True)
print(df.head())
print(len(df))
pause=input("Press Enter to continue")
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
#                                                                  INVERSION 
#
###########################################################################################################################################################
pinger=df.Ping_Number.unique()
print(pinger,type(pinger))
z_inv=[]
#prom_ping=[]
#dif_depth=[]
for ping in pinger:
    df_filter=df[df.Ping_Number==ping]
    X_p=df_filter['Footprint_X'].values
    Z_p = df_filter['Footprint_Z'].values
    beam=df_filter['Beam'].values
    mline=LS_solution_line(beam,Z_p)
    
    Z_inv_line=mline[0]+(mline[1]*beam)
    
    #plt.figure()
    #plt.plot(beam,Z_p,'b*',label='DATA')
    #plt.plot(beam, Z_inv_line,'r-',label='MODEL')
    #plt.legend(loc='best')
    #plt.show()   
    
    for z in Z_inv_line:
        z_inv.append(z)

df=df.assign(Z_inv_pinger=z_inv)
print(df.head())
print(len(df))

pause=input("Press Enter to continue")
#pause=input("Press Enter to continue")

print(pinger)
window_size=4000; # good test were run from 600
ZZ=[]
for ping in pinger:
    slice_df=df[(df.Ping_Number>ping-window_size)&(df.Ping_Number<ping+window_size)]#&(df.Beam>5)&(df.Beam<400)]
    
    slice_one=df[(df.Ping_Number==ping)].copy()#&(df.Beam>5)&(df.Beam<400)].copy()  
    
    
    model=LinearRegression().fit(slice_one.Beam.values.reshape((-1,1)),slice_one.Footprint_Z.values)
    
    model_1=LinearRegression().fit(slice_df.Beam.values.reshape((-1,1)),slice_df.Footprint_Z.values)
    
    slice_one['Z_T']=model.predict(slice_one.Beam.values.reshape((-1,1)))
    slice_one['Z_TT']=model_1.predict(slice_one.Beam.values.reshape((-1,1)))

    slice_one['Z_corr']=slice_one.Footprint_Z-slice_one.Z_T+slice_one.Z_TT
    
    Zcorr=slice_one['Z_corr'].values
    
    df_out=pd.concat([df_out,slice_one],ignore_index=True)



print(df_out.head())

        
# # ASSIGN THE NEW VALUES TO THE DATAFRAME   
# print(min(z_inv_stack),max(z_inv_stack))     
# df=df.assign(Z_inv_stack=z_inv_stack)
# print(df.head())
# pause=input("Press Enter to continue") 
# print("fianl step")
# #########################################################################################################################################################
# #########################################################################################################################################################

Yf=df_out['Footprint_Y'].values
Xf=df_out['Footprint_X'].values
Z_initial=df_out['Footprint_Z'].values
Zcorr=df_out['Z_corr'].values
# Z_ping=df['Z_inv_pinger'].values
# Z_stack_ping=df['Z_inv_stack'].values
# Z_corr=((Z_initial*-1.0)+(Z_ping*-1.0)-(Z_stack_ping*-1.0))*-1.0
# df=df.assign(Z_corr=Z_corr)    

# # Step 1: Filter the data
# # Filter the data to remove any NaN values
# # crate a surface from x y z
# # Create a meshgrid
# # Interpolate Z values based on scattered data
# # You can use any interpolation method you prefer here, for example, linear interpolation
# # SaveX, Y, Z to file
# # Plot the surfac
# #yi=np.unique(Y)
xi = np.linspace(min(Xf), max(Xf), 1000)
yi = np.linspace(min(Yf), max(Yf), 1000)
xi, yi = np.meshgrid(xi, yi)

# # Step 3: Interpolate the Z values onto the grid
# print("Grdidding data .......")
#z_initial_F = griddata((Xf, Yf), Z_initial, (xi, yi), method='cubic')
z_final = griddata((Xf, Yf), Zcorr, (xi, yi), method='cubic')


vmin = np.min(Z_initial)
vmax = np.max(Z_initial)
print ('min {0}, max {1}'.format(vmin, vmax))
# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
# # FILTERING KERNELS
# plt.figure()
# plt.scatter(Xf, Yf, c=Z_initial, s=2)
# plt.axis("scaled")
# plt.colorbar()
# plt.show()


plt.figure()
plt.scatter(Xf, Yf, c=Zcorr, s=2)
plt.axis("scaled")
plt.colorbar()
plt.show()



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#median = scipy.signal.medfilt2d(z_final,11)

# plt.imshow(z_initial_F,vmin=vmin,vmax=vmax)
# plt.colorbar()
# plt.title("Original")
# plt.show()


plt.imshow(z_final,vmin=vmin,vmax=vmax)
plt.colorbar()
plt.title("LS regression solution")
plt.show()




# plt.imshow(z_final,vmin=vmin,vmax=vmax)
# plt.colorbar()
# plt.title("LS solution")
# plt.show()



#plt.imshow(median,vmin=vmin,vmax=vmax)
#plt.colorbar()
#plt.title("Mean")
#plt.show()


end = time.time()
elapsed_time=end - start

print('Execution time:', elapsed_time/60, 'minutes')
