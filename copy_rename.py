import numpy as np 
import os
import glob
import shutil
##################################################################################################################
# FUNCTION DEFINITION
# FIND FILES WITH CERTAIN EXTENSION AND CERTAIN PATH 
def get_files(path_in, extension):
    files = glob.glob(path_in + '/*'+ extension)
    return files
# list the found files
def print_files(file):
    files=[]
    for i in range(len(file)):
        print(file[i])
        files.append(file[i])
    return files

# save the files in a list for 
def save_only_filenames(files,path_in):
    k=len(path_in)+1
    m=k+16
    only_file_names=[]
    for i in files:
        #print(i[])
        only_file_names.append(i[k:m]+extension)
    return only_file_names    


# add the new path to the file name
def new_Path(only,path_out):
    new_path=[]
    for i in only:
        new_path.append(path_out+"\\"+i)
    return new_path


# finally move the archives to the new path
def copy_rename(files,newfolder):
    for i in range(len(files)):
        shutil.copy(files[i], newfolder[i])
        print("File {} has been copied to {}".format(files[i],newfolder[i]))
    return
##################################################################################################################
print("\n")
print("\n")
print("\n")
print('########################################################################################################')
print('#                                                                                                      #')
print('#                  Indique la extension de los archivos que necesita mover                             #')
#extension=input("#Enter the extension of the file in format e.g. .sgy:         \n                              ")
extension=".sgy"
print('#                                                                                                      #')
print('########################################################################################################')
print("\n")
print("\n")

print('########################################################################################################')
print('#                                                                                                      #')
print('#                  Indique el path donde estan los archivos que se quieren mover:                      #')
#path_in=input("Enter the path of the folder where the files are located: e.g. /home/user/folder or D:\SEGY\SEGY_Files\sbp\yoti \n")
path_in='D:\SEGY\SEGY_Files\sbp\yoti'
print('#                                                                                                      #')
print('########################################################################################################')

print("\n")
print("\n")
file=get_files(path_in, extension)

print("Numero de archivos encontrados: {} archivos  ".format(len(file)))
print("\n")
list_answer=input("Quiere que se imprima la lista de archivos encontrados?:  y/n  \n")
print("\n")
print("\n")
lista_archivos=print_files(file)
print("\n")
print("\n")
only=save_only_filenames(lista_archivos,path_in)



print(only)
print('########################################################################################################')
print('#                                                                                                      #')
print('#                       Indique el path donde quiere los archivos:                                     #')
#path_out=input("Enter the path of the folder where the files are located: e.g. /home/user/folder")
path_out='D:\SEGY\SEGY_Files\sbp\yoti\test'
print('#                                                                                                      #')
print('########################################################################################################')
#print(file)

newfolder=new_Path(only,path_out)

copy_rename(lista_archivos,newfolder)



































print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("\n")
print("Successfully Created File and Rename")
print("\n")
print("\n")
print('########################################################################################################')
print('#               ooooo  o    o    ooooo  o  o                  o     o  oooooo    o     o               #')
print('#               o      o    o    o      o o                   o   o   o    o    o     o                #')
print('#               oo     o    o    o      oo                     o o    o    o    o     o                #')
print('#               o      o    o    o      o o                    o     o    o    o     o                 #')
print('#               o      oooooo    ooooo  o  o                  o     oooooo    ooooooo                  #')
print('#                                                                                                      #')
print('########################################################################################################')