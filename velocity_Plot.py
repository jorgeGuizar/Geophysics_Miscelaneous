import numpy as np 
import matplotlib.pyplot as plt

def Read_velocity_File(FILENAME):
    file=np.loadtxt(FILENAME,delimiter='\t',skiprows=1)
    return file

filepath="D:\SEGY\SEGY_Files\OPEX\VELOCIDADES\RMS_NoSmooth\ETKAL-202_L11_P0_250X250M_V3.txt"

file=Read_velocity_File(filepath)
print(file)
cdps=np.unique(file[:,0])
fig, ax1 = plt.subplots()
for i in cdps:
    data=file[file[:,0]==i]
    ax1.plot(data[:,2],data[:,1],label='CDP: '+str(i))
    ax1.invert_yaxis()
    plt.legend(loc='best')
ax1.set_xlabel('Velocity (m/s)')
ax1.set_ylabel('Time (m/s)')
ax1.invert_yaxis()
plt.title('Velocity Plot L11_P0')
plt.grid(True)
plt.show()