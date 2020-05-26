# -*- coding: utf-8 -*-
"""
Created on Sun May 24 21:59:36 2020

@author: LENOVO
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
plt.close("all")
#temp vs profu
filename='serie3.txt'
import scipy.ndimage
#############################################################################
# Function definition

def dataload(filename):   
    data = np.loadtxt( filename )
    return data
def vectorshape(data):
    lines,columns=data.shape
    return lines,columns

def XY_DEF(data):
    TEMP=data[:,0].copy()
    DEPTH=data[:,1].copy()
    return TEMP,DEPTH

def exp_smooth(Time_Series,a):
    N=len(Time_Series);
    Smooth_TS=np.zeros(N);
    Smooth_TS[0]=Time_Series[0];
    for  n in range(1,N):
        Smooth_TS[n]=a*Time_Series[n]+((1-a)*Smooth_TS[n-1])
        
    return Smooth_TS
    

############################################################################
# Funciton application
data=dataload(filename)
TEMP,DEPTH=XY_DEF(data)
a = 0.2;
yh=exp_smooth(TEMP,a)

############################################################################
#FFT
dt=0.5
Fs=1/dt

    
#############################################################################
#def varas(data):
 #   x

#print(data.shape,type(data))
plt.plot(TEMP,DEPTH*-1,label='Raw')
plt.plot(yh,DEPTH*-1,label='Smoothed')
plt.gca().invert_yaxis()
plt.title("Temperature vs Depth")
plt.xlabel("Temperature $^{°}C$")
plt.ylabel("Depth [m]")
plt.grid(True)
plt.legend(loc='best')

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

# plot time signal:
axes[0, 0].set_title("Signal")
axes[0, 0].plot(TEMP, DEPTH, color='C0')
axes[0, 0].plot(yh, DEPTH, color='C1')
plt.gca().invert_yaxis()
axes[0, 0].set_xlabel("Time")
axes[0, 0].set_ylabel("Amplitude")

# plot different spectrum types:
axes[1, 0].set_title("Magnitude Spectrum")
axes[1, 0].magnitude_spectrum(yh, Fs=Fs, color='C1')
axes[1, 0].magnitude_spectrum(TEMP, Fs=Fs, color='C4')

axes[1, 1].set_title("Log. Magnitude Spectrum")
axes[1, 1].magnitude_spectrum(yh, Fs=Fs, scale='dB', color='C1')
axes[1, 1].magnitude_spectrum(TEMP, Fs=Fs, scale='dB', color='C2')

axes[2, 0].set_title("Phase Spectrum ")
axes[2, 0].phase_spectrum(yh, Fs=Fs, color='C2')
axes[2, 0].phase_spectrum(TEMP, Fs=Fs, color='C3')

axes[2, 1].set_title("Angle Spectrum")
axes[2, 1].angle_spectrum(yh, Fs=Fs, color='C2')
axes[2, 1].angle_spectrum(TEMP, Fs=Fs, color='C3')

axes[0, 1].remove()  # don't display empty ax

fig.tight_layout()
plt.show()

