import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys


def Fold_Plot(file, header,title):
    if header=="Yes":
        data=np.loadtxt(file,delimiter='\t',skiprows=1)
        plt.scatter(data[:,0],data[:,1],c=data[:,2],cmap='jet')
        plt.grid(True)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('CDP X')
        plt.ylabel('CDP Y')
        plt.show()
    elif header=="No":
        data=np.loadtxt(file,delimiter='\t',skiprows=0)
        plt.scatter(data[:,0],data[:,1],c=data[:,2],cmap='jet')
        plt.grid(True)
        plt.colorbar()
        plt.title(title)
        plt.xlabel('CDP X')
        plt.ylabel('CDP Y')
        plt.show()