import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

from seismic_attributes import *


#################################################################################################

#fig, ax = plt.subplots(figsize=(16,10))

def plot_it(ax, data, vmin, vmax, title, cmap, ticks, ticklabels, alpha=1.0, interpolation='bilinear'):
    """_summary_

    Args:
        ax (_type_): _description_
        data (_type_): aray to plot must be the trasponse 
        vmin (_type_): min of data range that colormap uses
        vmax (_type_): max of data range that colormap uses
        title (_type_): title of the plot
        cmap (_type_): colormap for the plot
        ticks (_type_): _description_
        ticklabels (_type_): _description_
        alpha (float, optional): _description_. Defaults to 1.0.
        interpolation (str, optional): _description_. Defaults to 'bilinear'.
    """
    #fig, ax = plt.subplots(figsize=(16,10))
    im = ax.imshow(data.T, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto', alpha=alpha, interpolation=interpolation)
    ax.set_title(title)
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    cbar = fig.colorbar(im, 
                        ax=ax,
                        ticks=ticks,
                        orientation='horizontal',
                        fraction=0.03, pad=0.03)
    cbar.ax.set_xticklabels(ticklabels)
    plt.show()





#################################################################################################

dt=.004

seismic_npy = np.load(r'C:\Users\LEGA\Documents\Geofisica\MB\kerry3d.npy')

print(seismic_npy.shape)


rec_len=((int(seismic_npy.shape[-1]))-1)*dt*1000
print("INLINES: {}".format(seismic_npy.shape[0]))
print("CROSSLINES: {}".format(seismic_npy.shape[1]))
print("SAMPLES: {}".format(seismic_npy.shape[-1]))
print("RECORD_LENGTH:  {} ms".format(rec_len))


seismic_npy_SubCube = seismic_npy[143,:,:]

vmin, vmax = np.min(seismic_npy_SubCube), np.max(seismic_npy_SubCube)
vrng = max(abs(vmin), vmax)

hilbert_envelope=instantaneous_amplitude(seismic_npy_SubCube)

hilbert=Hilbert_data(seismic_npy_SubCube)
#plt.imshow(seismic_npy[143,:,:].T,cmap='gray', interpolation='bicubic')
#plt.figure(figsize=(6, 10))
#plt.title("Seismic Data")
#plt.imshow(seismic_npy_SubCube.T,cmap='gray', interpolation='bicubic')
#plt.show()

fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,seismic_npy_SubCube, -vrng/5, vrng/5, 'Seismic', 'seismic', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'],interpolation="bicubic")

fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,seismic_npy_SubCube, -vrng/5, vrng/5, 'Hilbert', 'seismic', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'],interpolation="bicubic")






plt.figure(figsize=(6, 10))
plt.title("Envelope Attribute")
plt.imshow(hilbert_envelope.T, interpolation='bicubic')
plt.show()


fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,hilbert_envelope, -vrng/5, vrng/5, 'Envelope', 'viridis', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'],interpolation="bicubic")

#HIL=quadrature(seismic_npy_SubCube)
#plt.figure(figsize=(6, 10))
#plt.title("HILBERT")
#plt.imshow(HIL.T,cmap='gray')
#plt.show()


PHASE=instantaneous_phase(seismic_npy_SubCube)

fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,PHASE, -vrng/5, vrng/5, 'Instantaneous Phase', 'twilight_shifted', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'],interpolation="bicubic")



freq=instantaneous_frequency(seismic_npy_SubCube,dt=0.004)

vmin, vmax = np.min(freq), np.max(freq)
vrng = max(abs(vmin), vmax)

fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,freq, -vrng/5, vrng/5, 'Instantaneous frequency', 'twilight_shifted', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'],interpolation="bicubic")

#plt.figure(figsize=(6, 10))
#plt.title("PHASE")
#plt.imshow(HIL.T,cmap='twilight_shifted', interpolation='none')
#plt.show()

#print(HIL.shape)



