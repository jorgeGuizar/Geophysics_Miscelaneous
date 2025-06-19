import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir 
from os.path import isfile,join


#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN4_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN4_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL1_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL1_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN1_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN1_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL7_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL7_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN2_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN2_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL2_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL2_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN12_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN12_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL3_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL3_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN3_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN3_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL4_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL4_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL8_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL8_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL5_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL5_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN14_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN14_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL9_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL9_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN15_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN15_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL6_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL6_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN5_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN5_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P1_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P1_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P1_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P1_SFSG.txt',delimiter='\t',skiprows=1)


#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P1_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P1_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN16_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN16_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL11_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL11_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN21_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN21_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN7_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN7_P0_SFSG.txt',delimiter='\t',skiprows=1)

#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN9_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN9_P0_SFSG.txt',delimiter='\t',skiprows=1)
#data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN18_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN18_P0_SFSG.txt',delimiter='\t',skiprows=1)

data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN10_P0_CFCG.txt',delimiter='\t',skiprows=1)
data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN10_P0_SFSG.txt',delimiter='\t',skiprows=1)

plt.plot(data[:,0],data[:,1],label='CFCG')
plt.plot(data2[:,0],data2[:,1],label='SFSG')
plt.grid(True)
plt.xlabel('Frequency Hz')
plt.ylabel('Amplitude dB')
plt.title('Espectro de Amplitud dB \n IN_10_P0')
plt.legend(loc='best')
plt.show()

