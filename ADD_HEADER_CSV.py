import csv
import os
import numpy as np 
import matplotlib.pyplot as plt
from UTILS import *
import time
import progressbar 
from time import sleep
from progressbar import progressbar   

#################################################################################################################
#
#  DO NOT MODIFY BELOW THIS LINE
#
#################################################################################################################
# REVISA SI EXISTE EL DIRECTORIO DE SALIDA Y SI NO LO CREA 

def agregar_encabezado_CSV_2(archivo_in, header,archivo_out):
    with open(header,'r') as file:
        header=file.read()
    
    with open(archivo_in, 'r') as file:
        contenido_original = file.read()

    with open(archivo_out, 'w') as file:
        file.write(header.strip() + '\n' + contenido_original.lstrip())


####################################################################################################################
print("\n")
print   (" Script executing ...")  
print("\n")
print   (" INSTRUCTIONS:  \n")  
print('########################################################################################################')
print('#                                                                                                      #')
print('#                  Indique el path donde estan los archivos SEGY:                                      #')
print('#                       FORMAT:            D:\SEGY\EBCDICs                                             #')
print('# 1.- INDIQUE LA RUTA DE LA CARPETA DONDE DESEA BUSCAR LOS ARCHIVOS .CSV A AGREGAR EL ENCABEZADO       #')
print('#                          PATH_IN                                                                     #')
print('# 2.- INDIQUE LA RUA DONDE VA A QUERER GUARDAR LOS NUEVOS ARCHIVOS  EN FORMATO .CSV                    #')
print('#                          PATH_OUT                                                                    #')
print('# 3.- INDIQUE LA RUTA DEL ARCHIVO QUE CONTENGA SU HEADER EN FORMATO .TXT                               #')
print('#                            header                                                                    #')
print('########################################################################################################')
print("\n")
print("\n")
#################################################################################################################
# DEFINICION DE PATHS Y HEADER
# MODIFIQUE AQUI LOS PATHS Y EL HEADER
PATH_IN=r"C:\Users\LEGA\Documents\CSV_HEADER"
PATH_OUT=r"C:\Users\LEGA\Documents\CSV_HEADER\Modified"
header=r"C:\Users\LEGA\Documents\CSV_HEADER\header.txt"
#################################################################################################################
# DEFINICION DE LA EXTENSION  SI EL AARCHIVO ES OTRO  CAMBIE EL STRING USANDO EL MISMO FORMATO QUE EL EJEMPLO
string_inside_file="\.CSV"



#################################################################################################################
#
#  DO NOT MODIFY BELOW THIS LINE
#
#################################################################################################################
# REVISA SI EXISTE EL DIRECTORIO DE SALIDA Y SI NO LO CREA 
if not os.path.exists(PATH_OUT):
    # Create the directory
    os.makedirs(PATH_OUT)
    print("Directory created successfully!")
else:
    print("Directory already exists!")

print("\n")
print   (" Reading Files ...") 
print   (" Reading Files ...") 
print   (" Reading Files ...") 
print("\n")
##################################################################################################################
##################################################################################################################
file_list=get_file_list(PATH_IN)
#PATH_IN="D:\SEGY\EBCDICs"
print(" Folder contents:  \n \n")
print(file_list)
print("\n")
print("Number of files found: {} Files ".format(len(file_list)))

##################################################################################################################
##################################################################################################################    
print("\n")
print("\n")
print("\n")
print('########################################################################################################')
print('#                                                                                                      #')
print('#                  Type the  string contained in the files chnage the header to (e.g.  .sgy)                           #')
#extension=input("#Enter the extension of the file in format e.g. .sgy:         \n                              ")
print('#                                                                                                      #')
print('########################################################################################################')
print("\n")
print("\n")
#string_inside_file=input("String contained in the file (e.g. WGS84):  \n \n")
print('The extension is {}'.format(string_inside_file))
filtered_list=filter_list(file_list,string_inside_file)
print("\n")
print("\n")
print(" SEGY files to count traces:")
print("\n")
print(filtered_list)
print("\n")
print("Number of files found: {} Files ".format(len(filtered_list)))


print("\n")
#print("Quieres agregar un sufijo a los archivos? (Y/N)")
sufijo=input("Quieres agregar un sufijo a los archivos? (YES/NO) \n")
for archivo in filtered_list:
    if sufijo=="YES" or sufijo=="yes" or sufijo=="Y" or sufijo=="y":
        file_out=PATH_OUT+"\\"+archivo[len(PATH_IN)+1:-4]+"_MODIFIED.csv"
        agregar_encabezado_CSV_2(archivo, header,file_out)
    else:
    #print(archivo)
        file_out=PATH_OUT+"\\"+archivo[len(PATH_IN)+1:]
    #print(file_out)
        agregar_encabezado_CSV_2(archivo, header,file_out)
print("\n")
print("\n")

for i in progressbar(range(100)):
    sleep(0.02)
print("\n")
print('#                                                                                                      #')
print('#                            LISTO:  ARCHIVOS MODIFICADOS  :) :) NOT SMILING?   FUCK U THEN            #')  
print("\n")   