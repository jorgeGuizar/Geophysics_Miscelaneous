import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import FormatStrFormatter
import tkinter
import random
from XLS_CSV_libaries import *
#import sys
#sys.modules[__name__].__dict__.clear()
path=r'D:\demar_3.nmp\demar_3\images_GASODUCTO_BN_16Ø _280425'
isExist = os.path.exists(path)
if not isExist:
   os.makedirs(path)#Create a new directory because it does not exist

   print("The new directory is created!: {}".format(path))
######################################################################################################################################################
######################################################################################################################################################
################################################## LIST OF DATA SHEETS#################################################################################
# archivo de prpofundidades
# Obtain the list of sheets in the excel file and filter them
file_pathprofundidades=r'C:\Users\LEGA\Downloads\TABLA-5MTS OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-AXANAB-C.xlsx'
xl = pd.ExcelFile(file_pathprofundidades)
lista_segmentos=xl.sheet_names
sheets_list=list(filter(lambda k: 'OGSD' in k, xl.sheet_names))
print("Lista de Sheet original")
print(xl.sheet_names)
print('NUMBER OF SHEETS {} \n'.format(len(sheets_list)) )
print(sheets_list)
######################################################################################################################################################


#xls = pd.read_excel(r'C:\Users\LEGA\Documents\PROFUNDIDADES OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-AXANAB-C.xlsx', sheet_name=['0+000-0+500', \
#    '0+501-1+000', '1+001-1+500', '1+501-2+000', '2+001-2+500', '2+501-3+000', '3+001-3+500', '3+501-4+000', '4+001-4+500', '4+501-5+000', \
#    '5+001-5+500', '5+501-6+000', '6+001-6+500', '6+501-7+000', '7+001-7+500', '7+501-8+000', '8+001-8+500', '8+501-9+000', '9+001-9+500', '9+501-10+000', '10+001-10+213'])
xls = pd.read_excel(file_pathprofundidades, sheet_name=sheets_list, skiprows=0, header=0, index_col=None, usecols="A:Q", engine='openpyxl')
df=xls['OGSD 20ØX10.4 KM'].copy()
print(df.columns)
input("Press Enter to continue...") # Pause for user input
a = np.arange(0.500, 10.100, .500)+.001
lim_inferior = np.concatenate(([0], a))
lim_superior = np.arange(0.500, 10.600, .500)
np.set_printoptions(precision=3)
k=0
split_1=1# indicator to split into sheets ina dataframe 
if split_1==0:
    with pd.ExcelWriter('monthly_report.xlsx', engine='openpyxl') as writer:
        for i in range(len(lim_inferior)):
            #filtered_df = df[(df['KP\n[km]'] > a_with_zero[i]) and (df['KP\n[km]'] < a_with_zero[i]+.500)]
            df4=df[(df['KP\n[km]'] >= lim_inferior[i]) & (df['KP\n[km]'] <= lim_superior[i] )].copy()
            print(len(df4))
            sheet=f'{lim_inferior[i]}-{lim_superior[i]}00'
            df4.to_excel(writer, sheet_name=sheet, index=False)
        
xls_2 = pd.read_excel(file_pathprofundidades, sheet_name=sheets_list, skiprows=0, header=0, index_col=None, usecols="B,C,D,K,L,M,Q", engine='openpyxl')
df_2=xls_2['OGSD 20ØX10.4 KM'].copy()   
print(df_2.columns) 

if split_1==0:
    with pd.ExcelWriter('Kps_split_7_final.xlsx', engine='openpyxl') as writer:
        for i in range(len(lim_inferior)):
            df_3=df_2[(df_2['KP\n[km]'] >= lim_inferior[i]) & (df_2['KP\n[km]'] <= lim_superior[i] )].copy()
            df_3.reset_index(drop=True, inplace=True)
            #print(df_3)
            long=len(df_3)
            indx=int(np.round(long/4))
            #print(len(df_3)) 
            
            df_final1= df_3.iloc[0:(indx*1)]
            df_final2 = df_3.iloc[(indx*1):(indx*2)]
            df_final3 = df_3.iloc[(indx*2):(indx*3)]
            df_final4 = df_3.iloc[(indx*3):]
            
            df_final1.reset_index(drop=True, inplace=True)
            df_final2.reset_index(drop=True, inplace=True)
            df_final3.reset_index(drop=True, inplace=True)
            df_final4.reset_index(drop=True, inplace=True)
            
            df_concat = pd.concat([df_final1, df_final2,df_final3,df_final4], axis=1)
            sheet=f'{lim_inferior[i]}-{lim_superior[i]}00'
            df_concat.to_excel(writer, sheet_name=sheet, index=False)


#B,c,d,k,l, mQ



