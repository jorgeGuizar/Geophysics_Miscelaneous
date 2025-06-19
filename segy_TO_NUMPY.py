from __future__ import print_function
import os
import sys
import segyio
import time
from obspy.io.segy.segy import _read_segy, SEGYBinaryFileHeader
from obspy import read
import numpy as np
import wget

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Only GPU 1 is visible to this code
#url='http://s3.amazonaws.com/open.source.geoscience/open_data/newzealand/Taranaiki_Basin/Keri_3D/Kerry3D.segy'
#filename=wget.download(url)
filename =r'C:\Users\LEGA\Documents\Geofisica\MB\Kerry3D.segy'
#
t0=time.time()
segy = _read_segy(filename)
print('--> data read in {:.1f} sec'.format(time.time()-t0))


binary_file_header = segy.binary_file_header
print("\nbinary_file_header:\n", binary_file_header)

textual_file_header = segy.textual_file_header
print("\ntextual_file_header:\n", textual_file_header)

data_encoding=segy.data_encoding
print("\ndata_encoding:\n",data_encoding)
endian=segy.endian
print("\nendian:\n", endian)
file=segy.file
print("\nfile:\n", file)
classinfo = segy.__class__
print("\nclassinfo:\n", classinfo)
doc = segy.__doc__
print("\ndoc:\n", doc)
ntraces=len(segy.traces)
print("\nntraces:\n", ntraces)
size_M=segy.traces[0].data.nbytes/1024/1024.*ntraces
print("\nsize:\n\t", size_M,"MB")
print("\t", size_M/1024, "GB")


#Read use read()
t0=time.time()
print('sgy use read:')
stream = read(filename)
print('--> data read in {:.1f} min'.format((time.time()-t0)/60))

print(stream)
print('\n')


i = 0 
trace_i_header = stream[i].stats.segy.trace_header
# print(trace_i_header)
print(stream[i].stats)

i = 0
trace_i_header = stream[i].stats.segy.trace_header
print("Crossline {}".format(trace_i_header.ensemble_number))# crossline number"
print("Coordinate X Source {}".format(trace_i_header.source_coordinate_x))
print("Coordinate Y Source {}".format(trace_i_header.source_coordinate_y))
print(trace_i_header.source_energy_direction_mantissa)
print("Inline {}".format(trace_i_header.source_energy_direction_exponent)) # inline number


# inlines and crosslines
il=[]
xl=[]
for i in range(len(stream)):
    trace_i_header = stream[i].stats.segy.trace_header
    il.append(trace_i_header.source_energy_direction_exponent)
    xl.append(trace_i_header.ensemble_number)

ilines = np.unique(il)
print(ilines)
print(len(ilines))

xlines = np.unique(xl)
print(xlines)
print(len(xlines))



from collections import Counter
t0=time.time()
counter = Counter(il)
print('Count in {:.1f} sec'.format(time.time()-t0))
print (sorted(counter.items()))


# this is a cube shape dataset.

seis_np = np.zeros((287,735,1252))
t0=time.time()
for i in range(210945): # total 1666070 traces in this dataset according to read() results shown above.
    tracei = stream[i]
    il=tracei.stats.segy.trace_header.source_energy_direction_exponent
    xl=tracei.stats.segy.trace_header.ensemble_number
    seis_np[il-510][xl-58] = tracei.data
print('--> data write in {:.4f} min'.format((time.time()-t0)/60))

import matplotlib.pyplot as plt
plt.figure(figsize=(20,16))
plt.title("inline 653")
plt.imshow(seis_np[143].transpose(), "gray") # inline 653 plot as: https://wiki.seg.org/wiki/Kerry-3D
plt.show()



import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
plt.imshow(seis_np[:,367,:].transpose(), "gray", aspect=0.3)
plt.show()


t0=time.time()
print('sgy save as npz:')
np.save(r"C:\Users\LEGA\Documents\Geofisica\MB\kerry3d_np",seis_np)
print('--> data save in {:.1f} min'.format((time.time()-t0)/60))
np.savez(r"C:\Users\LEGA\Documents\Geofisica\MB\kerry3d",seis_np)
print('--> data save in {:.1f} min'.format((time.time()-t0)/60))