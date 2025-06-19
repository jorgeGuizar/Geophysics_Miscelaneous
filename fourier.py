import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir 
from os.path import isfile,join

path2=r'D:/SEGY/txt/'

files=[f for f in listdir(path2) if isfile(join(path2,f))]
print(files)
paths=[]
for i in files:
    a=path2+i
    paths.append(a)
print(paths)


with open("D:/SEGY/txt/fold_all_yoti_3.txt","w") as new_file:
    for file in paths:
        data=np.loadtxt(file,delimiter='\t',skiprows=1)
        for i in range(len(data)):
            new_file.write(str(data[i,0])+'\t'+str(data[i,1])+'\t'+str(data[i,2])+'\n')
        #new_file.write('\t')

data=np.loadtxt('D:/SEGY/txt/fold_all_yoti_3.txt',delimiter='\t',skiprows=1)
plt.scatter(data[:,0],data[:,1],c=data[:,2],cmap='jet')
plt.grid(True)
plt.colorbar()

plt.title('Fold Yoti 2D')
plt.xlabel('CDP X')
plt.ylabel('CDP Y')

plt.show()


