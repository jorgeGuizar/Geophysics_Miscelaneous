import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
#import sys
#sys.modules[__name__].__dict__.clear()
path=r'D:\demar_3.nmp\demar_3\images'
isExist = os.path.exists(path)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(path)
   print("The new directory is created!: {}".format(path))

df = pd.read_excel(r'C:\Users\LEGA\Downloads\TABLAS DE PROFUNDIDADES_AKAL-NW.xlsx', skiprows=0, header=0, index_col=None, usecols="A:P", engine='openpyxl')
print(df.columns)

#decimation=input("")
df_1 = df[(df['KP\n[km]'] * 1000) % 5 == 0].copy()
    # Filter the DataFrame using the mask
    # Append the filtered DataFrame to the new DataFrame
    
    
df_1.to_csv(r'C:\Users\LEGA\Downloads\TABLAS_DE_PROFUNDIDADES_AKAL-NW_every5m.csv', index=False)