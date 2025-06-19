import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import pandas as pd 
import random
from scipy.interpolate import griddata
from datetime import datetime

####################################################################################################
grav=pd.read_csv(r'C:\Users\LEGA\Documents\Geofisica\MB\magnetometry\full\merged.INT',sep=' ')  


grav['Mag1_Mean'] = ''
grav['Mag1_Minus_Mean'] = ''



lines=grav['ROUTE'].unique()
for line in lines:
    grav_line=grav[grav['ROUTE']==line]
    grav_line['Mag1_Mean'] = grav_line['MAG1'].mean()
    grav_line['Mag1_Minus_Mean'] = grav_line['MAG1']-grav_line['Mag1_Mean']
    grav[grav['ROUTE']==line]=grav_line
    
####################################################################################################
print(lines)

print(grav.head())
####################################################################################################
# filter the data by it magnitude and filter the data by the polygon

grav_filt_gan=grav[grav['MAG1']>2000]

grav_filt=grav_filt_gan[(grav_filt_gan['SHIFT_LAT']>19.08461) & (grav_filt_gan['SHIFT_LAT']<19.12)]

print(grav_filt.info)

X=np.array(grav_filt['SHIFT_LON'].tolist())
Y=np.array(grav_filt['SHIFT_LAT'].tolist())
mag=np.array(grav_filt['MAG1'].tolist())
mag_mean=np.array(grav_filt['Mag1_Minus_Mean'].tolist())
print(np.min(mag), np.max(mag))
#print(X)
#print(len(X))

#samples=np.sort(random.sample(range(len(grav)),200000))

#X_sample=np.array([X[i] for i in samples])
#Y_sample=np.array([Y[i] for i in samples])
#mag_sample=np.array([mag[i] for i in samples])

#X_unique = np.sort(grav['SHIFT_LON'].unique())
#Y_unique = np.sort(grav['SHIFT_LAT'].unique())

# Step 2: Create a grid for interpolation
# Define the grid over which you want to interpolate
#xi=np.unique(X)
#yi=np.unique(Y)
xi = np.linspace(min(X), max(X), 1000)
yi = np.linspace(min(Y), max(Y), 1000)
xi, yi = np.meshgrid(xi, yi)

# Step 3: Interpolate the Z values onto the grid
zi = griddata((X, Y), mag, (xi, yi), method='cubic')
zi_m = griddata((X, Y), mag_mean, (xi, yi), method='cubic')






fig, ax = plt.subplots()
#ax.figure(figsize=(8, 6))
#plt.contourf(xi, yi, zi, levels=30, cmap='viridis')
#plt.contour(xi, yi, zi, levels=30, cmap='viridis')
plt.imshow(zi_m , cmap = 'jet' , interpolation = 'bicubic' , origin='lower',\
           aspect='auto', extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)],vmin=np.min(mag_mean), vmax=np.max(mag_mean)) 
#methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

           #aspect='equal',  extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)] ) 
plt.colorbar(label='Z Value')
#plt.scatter(X, Y, c=mag, cmap='viridis', edgecolor='k')  # Original points for reference
plt.title('2D Map from XYZ data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


fig, ax = plt.subplots()
#ax.figure(figsize=(8, 6))
#plt.contourf(xi, yi, zi, levels=30, cmap='viridis')
#plt.contour(xi, yi, zi, levels=30, cmap='viridis')
plt.imshow(zi , cmap = 'jet' , interpolation = 'bicubic' , origin='lower',\
           aspect='auto', extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)],vmin=np.min(mag), vmax=np.max(mag)) 
#methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
#           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
#           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

           #aspect='equal',  extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)] ) 
plt.colorbar(label='Z Value')
#plt.scatter(X, Y, c=mag, cmap='viridis', edgecolor='k')  # Original points for reference
plt.title('2D Map from XYZ data')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Step 4: Plot the 2D map
# fig, ax = plt.subplots()
# #ax.figure(figsize=(8, 6))
# #plt.contourf(xi, yi, zi, levels=30, cmap='viridis')
# #plt.contour(xi, yi, zi, levels=30, cmap='viridis')
# plt.imshow(zi_m , cmap = 'jet' , interpolation = 'bicubic' , origin='lower',\
#            aspect='auto', extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)],vmin=np.min(mag_mean), vmax=np.max(mag_mean)) 
# #methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
# #           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
# #           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

#            #aspect='equal',  extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)] ) 
# plt.colorbar(label='Z Value')
# #plt.scatter(X, Y, c=mag, cmap='viridis', edgecolor='k')  # Original points for reference
# plt.title('2D Map from XYZ data')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# chunks dependeito de los procesadores y grid de los bloques de datos
# 1. leer el archivo
# 2. filtrar los datos
# 3. interpolar los datos
# 4. graficar los datos

methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

# Fixing random state for reproducibility


fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})

for ax, interp_method in zip(axs.flat, methods):
    ax.imshow(zi_m , cmap = 'jet' , interpolation = interp_method , origin='lower',\
           aspect='auto', extent = [np.min(X), np.max(X), np.min(Y), np.max(Y)],vmin=np.min(mag_mean), vmax=np.max(mag_mean))
    ax.set_title(str(interp_method))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

plt.tight_layout()
plt.show()
