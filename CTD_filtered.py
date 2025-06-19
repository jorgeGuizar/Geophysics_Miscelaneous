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


df=pd.read_csv(r'C:\Users\LEGA\Downloads\SVP-01_AKAL-NW_V2.csv', header=0, )
print(df)

#print(df_FINAL)
# clean up the data
#df_final=df_final.drop()
# RENAME COLUMNS
df.rename(columns={"Unnamed: 1": "Depth(m)",'Depth(m)':  'Pressure (DBar)', \
    'Pressure (DBar)':'Temperature (DegC)','Temperature (DegC)':'Conductivity (mS/cm)','Conductivity (mS/cm)':'Salinity (PSU)', \
        'Salinity (PSU)':'Sound Velocity (MS-1)','Sound Velocity (MS-1)':'Density (kg/M3)','Density (kg/M3)':'vacio'}, inplace=True)
#df_FINAL = df.dropna()#subset=['Depth(m)'])
df.drop(['vacio'], axis=1, inplace=True)
print(df)
# drop  negative values of headers
df_final_1=df[(df['Pressure (DBar)'] > 0) & (df['Conductivity (mS/cm)'] > 0) & (df['Depth(m)'] > 0) \
    & (df['Sound Velocity (MS-1)']>1400) & (df['Temperature (DegC)']>20) & (df['Salinity (PSU)']>0) ].copy()
df_final_1=df_final_1.reset_index(drop=True)
print(df_final_1)
#input("Press Enter to continue...") # Pause for user input
#######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
# FIND THE INCREASING VALUES 
increasing_values = [df_final_1['Depth(m)'].iloc[0]]
print(increasing_values)
# Iterate through the values in the 'Depth(m)' column
for val in df_final_1['Depth(m)'].iloc[1:]:
    if val > increasing_values[-1]:
        increasing_values.append(val)
        
df_final_2_inc=df_final_1[df_final_1['Depth(m)'].isin(increasing_values)].copy()
#input("Press Enter to continue...")
#filtered_df = pd.DataFrame({'values': increasing_values})
#######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
# FIND THE DECREASING VALUES 
# Find the index of the first maximum value
max_idx = df_final_1['Depth(m)'].idxmax()
print(max_idx,df_final_1['Depth(m)'].iloc[max_idx])
#input("Press Enter to continue...") # Pause for user input
# Start from the max value and look forward for strictly decreasing values
start_value = df_final_1.loc[max_idx, 'Depth(m)']
print(start_value)
#input("Press Enter to continue...")
decreasing_values = [start_value]
print(decreasing_values)
#input("Press Enter to continue...")
#decreasing_values=[]
#decreasing_values=start_value
indices = [max_idx]
# Iterate through the values in the 'Depth(m)' column starting from the max value

for i in range(max_idx + 1, len(df_final_1)):
    current_value = df_final_1.loc[i, 'Depth(m)']
    print(current_value)
    if current_value < decreasing_values[-1]:
        decreasing_values.append(current_value)
        indices.append(i)      
        
        
df_final_2_dec=df_final_1[df_final_1['Depth(m)'].isin(decreasing_values)].copy()
df_final_2_inc=df_final_1[df_final_1['Depth(m)'].isin(increasing_values)].copy()

df_unique_dec= df_final_2_dec.drop_duplicates(subset=['Depth(m)'], keep='first')
df_unique_inc= df_final_2_inc.drop_duplicates(subset=['Depth(m)'], keep='first')

df_final_concat = pd.concat([df_unique_inc, df_unique_dec], axis=0)  

df_final_concat =df_final_concat.reset_index(drop=True)
print(df_final_concat)

#input("Press Enter to continue...") # Pause for user input
output_file= r'C:\Users\LEGA\Downloads\SVP-01_AKAL-NW_V2_FILTERED_norep.csv'
#df_final_2.to_csv(output_file_nosort, index=False)
df_final_concat.to_csv(output_file, index=False)