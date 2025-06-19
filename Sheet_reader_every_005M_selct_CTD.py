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

#df = pd.read_excel(r'C:\Users\LEGA\Downloads\TABLA_2.xlsx', skiprows=0, header=0, index_col=None, usecols="A:P", engine='openpyxl')
#df=pd.read_csv(r'C:\Users\LEGA\Downloads\SVP-01_ATOYATL-A.csv', skiprows=0, header=0)# index_col=None, usecols="A:I", engine='python')
df=pd.read_csv(r'C:\Users\LEGA\Downloads\SVP-01_ATOYATL-A.csv', header=0)
print(df.columns)
df_clean=df.dropna()
depth_min=df_clean['Depth(m)'].min()
depth_max=df_clean['Depth(m)'].max()

valores=[]
for i in range(int(depth_min), int(depth_max)+6, 5):
    valores.append(i)
print(valores)

depth = df_clean['Depth(m)'].values
#input("Press Enter to continue...") # Pause for user input
# Define the step size (every 5 units)
step = 4

# Find min and max Depth to determine range
min_val = np.floor(depth.min() / step) * step
max_val = np.ceil(depth.max() / step) * step
print(min_val, max_val)

# Generate target values (0, 5, 10, 15, ...)
targets = np.arange(min_val, max_val + step, step)

# Find the closest row for each target
filtered_rows = []
for target in targets:
    # Compute absolute difference from target
    diff = np.abs(depth - target)
    # Find the index of the closest value
    closest_idx = diff.argmin()
    filtered_rows.append(df_clean.iloc[closest_idx])
print(filtered_rows)
# Create a new DataFrame with filtered rows
filtered_df = pd.DataFrame(filtered_rows)
# Save to a new CSV
output_file = r'C:\Users\LEGA\Downloads\filtered_SVP-01_ATOYATL-A_closest_every_{}_Depth.csv'.format(step)
filtered_df.to_csv(output_file, index=False)