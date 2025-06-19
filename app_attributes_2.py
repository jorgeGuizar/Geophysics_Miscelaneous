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

t=np.arange(0,5,dt)*1000 

seismic_npy = np.load(r'C:\Users\LEGA\Documents\Geofisica\MB\kerry3d.npy')

print(seismic_npy.shape)


rec_len=((int(seismic_npy.shape[-1]))-1)*dt
print("INLINES: {}".format(seismic_npy.shape[0]))
print("CROSSLINES: {}".format(seismic_npy.shape[1]))
print("SAMPLES: {}".format(seismic_npy.shape[-1]))
print("RECORD_LENGTH:  {} ms".format(rec_len))


seismic_npy_SubCube = seismic_npy[143,:,:]

vmin, vmax = np.min(seismic_npy_SubCube), np.max(seismic_npy_SubCube)
vrng = max(abs(vmin), vmax)


fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,seismic_npy_SubCube, -vrng/5, vrng/5, 'Seismic', 'seismic', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'])



hilbert_envelope=instantaneous_amplitude(seismic_npy_SubCube)
fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax,hilbert_envelope, 0, vrng/5, 'Envelope', 'viridis', [0, vrng/5], [ '0', 'max'])


PHASE=instantaneous_phase(seismic_npy_SubCube)
fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax, PHASE, -np.pi, np.pi, 'Instantaneous Phase', 'twilight', [-np.pi, 0, np.pi], ['tau/2', '0', 'tau/2'])




HIL=quadrature(seismic_npy_SubCube)
fig, ax = plt.subplots(figsize=(16,10))
plot_it(ax, HIL, -vrng/5, vrng/5, 'Hilbert', 'seismic', [-vrng/5, 0, vrng/5], ['-ve', '0', '+ve'])



#freq_oldenburg=instantaneous_frequency(seismic_npy_SubCube, dt=.004, kind='claerbout')
#plt.figure(figsize=(6, 10))
#plt.imshow(freq_oldenburg.T, interpolation='bicubic', vmin=-10)
#plt.colorbar(shrink=0.75)
#plt.show()





semb = similarity(seismic_npy_SubCube, duration=0.036, dt=0.004, kind='gst')
plt.figure(figsize=(6, 10))
plt.title('Gradient Structure Tensor (GST) \n Coherence Attribute')
plt.imshow(semb.T, cmap='gray', interpolation='bicubic')
plt.show()


semb = similarity(seismic_npy_SubCube, duration=0.036, dt=0.004, kind='gersztenkorn')
plt.figure(figsize=(6, 10))
plt.title('Gradient Structure Tensor (GST) \n Coherence Attribute \n Gersztenkorn & Marfurt, 1999')
plt.imshow(semb.T, cmap='gray', interpolation='bicubic')
plt.show()

semb = similarity(seismic_npy_SubCube, duration=0.036, dt=0.004, kind='marfurt')
plt.figure(figsize=(6, 10))
plt.title('Gradient Structure Tensor (GST) \n Semblance based Coherence Attribute \n Marfurt, 1999')
plt.imshow(semb.T, cmap='gray', interpolation='bicubic')
plt.show()


bahorich = bahorich_coherence(seismic_npy_SubCube, 21)
plt.figure(figsize=(6, 10))
plt.title('Coherence \n Bahorich & Farmer 1995')
plt.imshow(bahorich.T, cmap='gray', interpolation='bicubic')
plt.show()



RMS_seismic=Rms_calc(seismic_npy_SubCube, tx=1, window=11)
plt.imshow(RMS_seismic.T,cmap='viridis', interpolation='bicubic')
plt.title('RMS Amplitude')
plt.show()