import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
#3import seaborn as sns
import scipy as scp
import mpl_toolkits.mplot3d
from scipy.interpolate import griddata


file=r"C:\Users\LEGA\Documents\Geofisica\MB\SonarData\KNOR_SON_IN_1_45L_xtf-CH12_xya.csv"



def import_data_full(file_name):
    # Import data from file
    #file_name = "DTM_IchA_0.5m.xyz"
    #file_name = "10-09-23_IchA_DTMc_0.5m.xyz"

    #df = pd.read_csv(file, delimiter=',', decimal='.',header=None, dtype=np.float32)
    df = pd.read_csv(file, delimiter=',', decimal='.', header=None, dtype=np.float32)

    # Create a matrix with the data
    x = data[:,0]
    y = data[:,1]
    z = data[:,3]

    return x, y, z

def meshing_sonar(x,y):
    x1 = np.unique(x)
    y1 = np.unique(y)
    X, Y = np.meshgrid(x1, y1,sparse=True)
    return X,Y

def Grid_Sonar_data(x,y,z,XX,YY):
    Z = griddata((x, y), z, (XX, YY), method='linear')
    return Z



    



x,y,z=import_data_full(file)
X,Y=meshing_sonar(x,y)
Z=Grid_Sonar_data(x,y,z,X,Y)

















