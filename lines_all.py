import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir 
from os.path import isfile,join


data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN4_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN4_P0_SFSG.txt',delimiter='\t',skiprows=1)
data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL1_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL1_P0_SFSG.txt',delimiter='\t',skiprows=1)
data3=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN1_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN1_P0_SFSG.txt',delimiter='\t',skiprows=1)
data4=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL7_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL7_P0_SFSG.txt',delimiter='\t',skiprows=1)
data5=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN2_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN2_P0_SFSG.txt',delimiter='\t',skiprows=1)
data6=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL2_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL2_P0_SFSG.txt',delimiter='\t',skiprows=1)
data7=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN12_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN12_P0_SFSG.txt',delimiter='\t',skiprows=1)
data8=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL3_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL3_P0_SFSG.txt',delimiter='\t',skiprows=1)
data9=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN3_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN3_P0_SFSG.txt',delimiter='\t',skiprows=1)
data10=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL4_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL4_P0_SFSG.txt',delimiter='\t',skiprows=1)
data11=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL8_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL8_P0_SFSG.txt',delimiter='\t',skiprows=1)

data12=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P0_SFSG.txt',delimiter='\t',skiprows=1)

data13=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL5_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL5_P0_SFSG.txt',delimiter='\t',skiprows=1)

data14=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN14_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN14_P0_SFSG.txt',delimiter='\t',skiprows=1)
data15=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL9_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL9_P0_SFSG.txt',delimiter='\t',skiprows=1)

data16=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN15_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN15_P0_SFSG.txt',delimiter='\t',skiprows=1)

data17=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL6_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL6_P0_SFSG.txt',delimiter='\t',skiprows=1)

data18=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN5_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN5_P0_SFSG.txt',delimiter='\t',skiprows=1)

data19=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P1_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P1_SFSG.txt',delimiter='\t',skiprows=1)
data=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P1_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN13_P1_SFSG.txt',delimiter='\t',skiprows=1)


data20=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P1_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P1_SFSG.txt',delimiter='\t',skiprows=1)

data21=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL10_P0_SFSG.txt',delimiter='\t',skiprows=1)

data22=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN16_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN16_P0_SFSG.txt',delimiter='\t',skiprows=1)

data29=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL11_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/XL11_P0_SFSG.txt',delimiter='\t',skiprows=1)

data30=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN11_P0_SFSG.txt',delimiter='\t',skiprows=1)

data31=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN21_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN21_P0_SFSG.txt',delimiter='\t',skiprows=1)

data32=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN7_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN7_P0_SFSG.txt',delimiter='\t',skiprows=1)

data33=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN9_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN9_P0_SFSG.txt',delimiter='\t',skiprows=1)
data34=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN18_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN18_P0_SFSG.txt',delimiter='\t',skiprows=1)

data35=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN10_P0_CFCG.txt',delimiter='\t',skiprows=1)
#data2=np.loadtxt('C:/Users/LEGA/Documents/Geofisica/MB/espectra/IN10_P0_SFSG.txt',delimiter='\t',skiprows=1)

plt.plot(data[:,0],data[:,1])
plt.plot(data2[:,0],data2[:,1])
plt.plot(data3[:,0],data3[:,1])
plt.plot(data4[:,0],data4[:,1])
plt.plot(data5[:,0],data5[:,1])
plt.plot(data6[:,0],data6[:,1])
plt.plot(data7[:,0],data7[:,1])
plt.plot(data8[:,0],data8[:,1])
plt.plot(data9[:,0],data9[:,1])
plt.plot(data10[:,0],data10[:,1])
plt.plot(data11[:,0],data11[:,1])
plt.plot(data12[:,0],data12[:,1])
plt.plot(data13[:,0],data13[:,1])
plt.plot(data14[:,0],data14[:,1])
plt.plot(data15[:,0],data15[:,1])
plt.plot(data16[:,0],data16[:,1])

plt.grid(True)
plt.xlabel('Frequency Hz')
plt.ylabel('Amplitude dB')
plt.title('Espectro de Amplitud dB \n CFCG')
plt.legend(loc='best')
plt.show()

