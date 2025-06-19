
import os
import re
import sys
import traceback
import collections
import shutil

def get_file_list(path):
    """

    Args:
        path (_type_): _description_

    Returns:
        _type_: _description_
    """
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def filter_list(file_list,pattern):
    """_summary_

    Args:
        file_list (_type_): _description_
        pattern (_type_): _description_

    Returns:
        _type_: _description_
    """
    filtered_list = [i for i in file_list if re.search(pattern, i)]
    return filtered_list


def copy_files(filtered_list,dest_path):
    """_summary_

    Args:
        filtered_list (_type_): _description_
        dest_path (_type_): _description_
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for file in filtered_list:
        shutil.copy(file,dest_path)
        
def rename_files(file_list,extension,):
    len_ext=len(extension)
    for i in range(len(file_list)):
        print("File {} has been renamed to {}".format(file_list[i],file_list[i][:-15]+file_list[i][-len_ext:]))
        #os.rename(file_list[i],file_list[i][:-15]+".sgy"+file_list[i][-4:])
        os.rename(file_list[i],file_list[i][:-15]+file_list[i][-len_ext:])
    return



def read_shift_file(file,st1,st2,dest_file):
    """This function reads a file and extracts the strings that are after st1 and st2. Then it writes the strings in a new file.
    Specifically  it reads the shift file for the SBP data, extracting the line number and shift and writing it in a new file.

    Args:
        file (.txt): file with information of every lines
        st1 (string of the  line): string  to look values into
        st2 (string of the shift): string  to look values into
        dest_file (.txt): output file in two columns with the line number and the shift
    """
    len_st1=len(st1)
    len_st2=len(st2)
    line1=[]
    line2=[]
    with open(file, 'r') as f:
        for line in f:
            if st1 in line:
                line1.append(line[len_st1:].strip())
                #line1.append(line[len_st1:-1])
            
            if st2 in line:
                line2.append(line[len_st2:].strip())
                #line2.append(line[len_st2:-1])
                
    #print(line1)  
    #print(line2)          
    with open(dest_file, 'w') as f:
        for archive,shift in zip(line1,line2):
            #f.write(archive+'\t\t\t\t'+shift+'\n')
            f.write('{0} \t\t {1} \n'.format(archive, str(shift)))
            #print(archive)
        print("File written successfully")
    
    f.close()
        
    return


def agregar_encabezado_CSV(archivo_in, header,archivo_out):
    
    """Summary:
    This script 
    
    
    
    
    """
    with open(header,'r') as file:
        header=file.read()
    
    with open(archivo_in, 'r') as file:
        contenido_original = file.read()

    with open(archivo_out, 'w') as file:
        file.write(header.strip() + '\n' + contenido_original.lstrip())