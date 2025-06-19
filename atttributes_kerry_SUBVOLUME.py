import numpy as np
import zipfile
import matplotlib.pyplot as plt
import bruges as bg
from scipy.signal import hilbert

seismic_npz = np.load(r'C:\Users\LEGA\Documents\Geofisica\MB\kerry3d.npz')
seismic_npy = np.load(r'C:\Users\LEGA\Documents\Geofisica\MB\kerry3d.npy')
    
print(type(seismic_npz))
print(type(seismic_npy))


seismic_npy_SubCube = seismic_npy[143]
def npz_headers(npz):
    """
    Takes a path to an .npz file, which is a Zip archive of .npy files.
    Generates a sequence of (name, shape, np.dtype).
    """
    with zipfile.ZipFile(npz) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                continue

            npy = archive.open(name)
            version = np.lib.format.read_magic(npy)
            shape, fortran, dtype = np.lib.format._read_array_header(npy, version)
            yield name[:-4], shape, dtype
            
            
#print(list(npz_headers("kerry3d.npz")))
#print(seismic_npy.shape)
#Hilbert=float(hilbert(seismic_npy_SubCube))
#Hilbert=bg.attribute.complex.hilbert(seismic_npy_SubCube)

#plt.figure(figsize=(6, 10))
#plt.title("Hilbert Attribute")
#plt.imshow(Hilbert.T, interpolation='bicubic')
#plt.show()


plt.figure(figsize=(6, 10))
plt.title("Seismic Data")
plt.imshow(seismic_npy_SubCube.T, interpolation='bicubic',cmap="seismic")
plt.show()




envelope=bg.attribute.complex.envelope(seismic_npy_SubCube)

plt.figure(figsize=(6, 10))
plt.title("Envelope Attribute")
plt.imshow(envelope.T, interpolation='bicubic')
plt.show()


phase = bg.attribute.complex.instantaneous_phase(seismic_npy_SubCube)

plt.figure(figsize=(6, 10))
plt.title("Instanteneous Phase Attribute")
plt.imshow(phase.T, cmap='twilight_shifted', interpolation='none')
plt.show()




inst_amp=bg.attribute.complex.instantaneous_amplitude(seismic_npy_SubCube)
plt.figure(figsize=(20, 10))
plt.title("Inst Amplitude Attribute")
plt.imshow(inst_amp.T, interpolation='bicubic')
plt.show()


semb = bg.attribute.similarity(seismic_npy_SubCube, duration=0.036, dt=0.004, kind='gst')

plt.figure(figsize=(6, 10))
plt.title("Similiarity Attribute")
plt.imshow(semb.T, cmap='gray', interpolation='bicubic')
plt.show()

discontonuity=bg.attribute.discontinuity(seismic_npy_SubCube, duration=0.036, dt=0.004, step_out=1, kind='marfurt', sigma=1)

plt.figure(figsize=(6, 10))
plt.title("Discontinuity Attribute Marfurt")
plt.imshow(discontonuity.T, cmap='gray', interpolation='bicubic')
plt.show()


discontonuity=bg.attribute.discontinuity(seismic_npy_SubCube, duration=0.036, dt=0.004, step_out=1, kind='gersztenkorn', sigma=1)

plt.figure(figsize=(6, 10))
plt.title("Discontinuity Attribute gersztenkorn")
plt.imshow(discontonuity.T, cmap='gray', interpolation='bicubic')
plt.show()