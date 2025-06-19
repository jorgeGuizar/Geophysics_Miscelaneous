import os
import re


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


PATH_IN=r'C:\Users\LEGA\Documents\HEADER\LOW'

PATH_OUT=r'C:\Users\LEGA\Documents\HEADER\LOW'

string_inside_file="\.txt"


file_list=get_file_list(PATH_IN)

file_list_filtered=filter_list(file_list,string_inside_file)


print(file_list_filtered)
print('\n')
print('Tienes {} archivos'.format(len(file_list_filtered)))










#file_in=r'C:\Users\LEGA\Documents\HEADER\LOW\L-02-P0L-header.txt'
#file_out=r'C:\Users\LEGA\Documents\HEADER\LOW\header\L-02-P0L-header.txt'
#PATH_OUT=r'C:\Users\LEGA\Documents\HEADER\LOW\header'
# REVISA SI EXISTE EL DIRECTORIO DE SALIDA Y SI NO LO CREA 







if not os.path.exists(PATH_OUT):
    # Create the directory
    os.makedirs(PATH_OUT)
    print("Directory created successfully!")
else:
    print("Directory already exists!")
    
for file_in in file_list_filtered:
    #print("File {} has been renamed to {}".format(file,file[:-15]+".sgy"+file[-4:]))
# Taking "gfg input file.txt" as input file 
# in reading mode 
    with open(file_in, "r") as input: 
        lines = input.readlines()[1:]
        # Creating "gfg output file.txt" as output 
        # file in write mode 
        with open(file_in, "w") as output: 
            # Writing each line from input file to 
            # output file using loop 
            for line in lines: 
                output.write(line)
    
    

    
        