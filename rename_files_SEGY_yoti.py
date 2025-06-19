import os
import re
import sys
import traceback
import collections
import shutil
from UTILS import *


print('########################################################################################################')
print('#                  INSTRUCCIONES                                                                       #')
# Path: copy_files.py
print("\n")
print   (" Script executing ...")  
print   (" Script executing ...")  
print   (" Script executing ...") 
print   (" Script executing ...")  
print   (" Script executing ...")  
print   (" Script executing ...")  
print("\n")
print('########################################################################################################')
print('#                                                                                                      #')
print('#                  Indique el path donde estan los archivos que se quieren Renombrar:                      #')
print('#                       FORMAT:            D:\SEGY\EBCDICs                                             #')
#path_in=input("Enter the path of the folder where the files are located: e.g. /home/user/folder or D:\SEGY\SEGY_Files\sbp\yoti \n")
#path_in='D:\SEGY\SEGY_Files\sbp\yoti'
print('#                                                                                                      #')
print('########################################################################################################')
print("\n")
print("\n")
#PATH_IN=input("Path (e.g. D:\SEGY\SEGY_Files\sbp\yoti):  \n \n")
#PATH_IN=r"D:\SEGY\SEGY_Files\MigratedStacks_FW\me"
PATH_IN=r"D:\SEGY\SEGY_Files\sbp\YOTI_SBP\YOTI"
print("\n")
print   (" Reading Files ...") 
print   (" Reading Files ...") 
print   (" Reading Files ...") 
print("\n")
##################################################################################################################
##################################################################################################################
file_list=get_file_list(PATH_IN)
#PATH_IN="D:\SEGY\EBCDICs"
print(" Do you want  to print all the files in the folder?  ")
print(" 1. Yes ")
print(" 2. No ")
print_files_raw=input("Enter 1 or 2:  \n \n")
if print_files_raw=="1":
    print("\n")
    print(file_list)
    print("\n")
    print("Number of files found: {} Files ".format(len(file_list)))
else:
    print("\n")
    print("Number of files found: {} Files ".format(len(file_list)))
    print("Ok, lets continue")
    print("\n")
print("\n")
##################################################################################################################
##################################################################################################################    
# rename files  

print("\n")
print("\n")
print("\n")
print('########################################################################################################')
print('#                                                                                                      #')
print('#                  Type the  string contained in the files that you want to move, can be \n            #')    
print('                either a string within the file or an extension (e.g.  .sgy)                           #')
#extension=input("#Enter the extension of the file in format e.g. .sgy:         \n                              ")
print('#                                                                                                      #')
print('########################################################################################################')
print("\n")
print("\n")
#string_inside_file=input("String contained in the file (e.g. WGS84):  \n \n")
string_inside_file="WGS84"
filtered_list=filter_list(file_list,string_inside_file)
print("\n")
print("\n")
print(" Do you want  to print all selected files?  ")
print(" 1. Yes ")
print(" 2. No ")
print_files_processed=input("Enter 1 or 2:  \n \n")
if print_files_processed=="1":
    print("\n")
    print(filtered_list)
    print("\n")
    print("Number of files found: {} Files ".format(len(filtered_list)))
else:
    print("\n")
    print("Number of files found: {} Files ".format(len(filtered_list)))
    print("Ok, lets continue")
    print("\n")
print("\n")


#a=int(input("Analize the files you want to rename and select the string that you want to remove from the file name, then press Enter ... (e.g. for a=9 then--> 10-09-23_IchA_DTMc_0.5m_xyz.csv --> 10-09-23_IchA_DTMc_0.5.csv  ) \n \n"))
a=".sgy"   
lista_names_new=[]

for i in file_list:
    lista_names_new.append(i+a)
    
print(lista_names_new)    
def rename_files_5(file_list,file_list_new):
    
    for old,new in zip(file_list,file_list_new):
        #os.rename(file_list[i],file_list[i][:-15]+".sgy"+file_list[i][-4:])
        os.rename(old,new)
    return

rename_files_5(filtered_list,lista_names_new)


