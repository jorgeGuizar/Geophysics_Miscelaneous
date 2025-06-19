import numpy as np
import zipfile
import matplotlib.pyplot as plt
import bruges as bg

seismic_npz = np.load(r'C:\Users\LEGA\Documents\Geofisica\MB\kerry3d.npz')
seismic_npy = np.load(r'C:\Users\LEGA\Documents\Geofisica\MB\kerry3d.npy')
    
print(type(seismic_npz))
print(type(seismic_npy))

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
            
            
print(list(npz_headers("kerry3d.npz")))
print(seismic_npy.shape)

envelope=bg.attribute.envelope(seismic_npy)

plt.figure(figsize=(6, 10))
plt.title("Envelope Attribute")
plt.imshow(envelope[143].T, interpolation='bicubic')
plt.show()


phase = bg.attribute.instantaneous_phase(seismic_npy)

plt.figure(figsize=(6, 10))
plt.title("Instanteneous Phase Attribute")
plt.imshow(phase[143].T, cmap='twilight_shifted', interpolation='none')
plt.show()


# semb = bg.attribute.similarity(seismic_npy, duration=0.036, dt=0.004, kind='gst')

# plt.figure(figsize=(6, 10))
# plt.title("Similiarity Attribute")
# plt.imshow(semb[143].T, cmap='gray', interpolation='bicubic')
# plt.show()