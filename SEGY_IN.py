import numpy as np 
import segyio
import matplotlib.pyplot as plt
from geophysic_UTILS import *
from UTILS import *
import time
#import progressbar

#SEGYFILE="D:\SEGY\SEGY_Files\MigratedStacks_FW\ICHALKIL_A\Time\SFSG\WGS84\ICHB4_A_L01_P0_PostSTM_Time_SFSG_WGS84.sgy"
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
print('#                  Indique el path donde estan los archivos SEGY:                      #')
print('#                       FORMAT:            D:\SEGY\EBCDICs                                             #')
#path_in=input("Enter the path of the folder where the files are located: e.g. /home/user/folder or D:\SEGY\SEGY_Files\sbp\yoti \n")
#path_in='D:\SEGY\SEGY_Files\sbp\yoti'
print('#                                                                                                      #')
print('########################################################################################################')
print("\n")
print("\n")
#PATH_IN=input("Path (e.g. D:\SEGY\SEGY_Files\sbp\yoti):  \n \n")

#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\SUPERB_modified\Processed_data_corrected"

#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\SUPERB_modified\Raw_Data_no_corrections_applied"

#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_A\Raw_Data_no_corrections_applied"
#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_A\Processed_data_Heave_Corrected_depth_corrected_as_appropriate"
#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\Processed_data_Heave_Corrected_depth_corrected_as_appropriate"
#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\Raw_Data_no_corrections_applied"
#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_C\Raw_Data_no_corrections_applied"
#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_C\Processed_data_Heave_Corrected_depth_corrected_as_appropriate"
#PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\Raw_Data_no_corrections_applied"#
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\1.0_PROCESSED_SMT'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\1.0_RAW_SMT_XY_ICHB'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_C\5.2_Raw_Data-no_corrections_applied'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_C\5.3_Processed_data-Heave-Corrected_depth_corrected_as_appropriate'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\1.0_PROCESSED_SMT_XY'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\1.0_RAW_SMT_XY_ICHSB'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_ICKALKIL_A\1.0_PROCESSED-SMT-XY\3.6_Block-6'

# 
# PATH_IN=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\Processed_data_Heave_Corrected_depth_corrected_as_appropriate"
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_POKOCH\5.2_Raw_Data-no-corrections-applied\5.0_BLOCK-4'

#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\5.3_Processed_data-Heave-Corrected_depth_corrected_as_appropriate'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\5.2_Raw_Data-no_corrections_applied'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_POKOCH\5.3 Processed_data-Heave-Corrected_depth_corrected_as_appropriate'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_POKOCH\5.2_Raw_Data-no-corrections-applied'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_ICHALKIL_A\5.3_Processed_data-Heave-Corrected_depth_corrected_as_appropriate\3.6_Block-6'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_ICHALKIL_A\5.3_Processed_data-Heave-Corrected_depth_corrected_as_appropriate\3.7_Cruces'

PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ENI\Processed-Lines-SMT_XY_depht'
#PATH_IN=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_ICHALKIL_A\5.3_Processed_data-Heave-Corrected_depth_corrected_as_appropriate\3.7_Cruces'

len_path_in=len(PATH_IN)
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
print('#                  Type the  string contained in the files that you want to move, can be \n            #')    
print('                either a string within the file or an extension (e.g.  .sgy)                           #')
#extension=input("#Enter the extension of the file in format e.g. .sgy:         \n                              ")
print('#                                                                                                      #')
print('########################################################################################################')
print("\n")
print("\n")
#string_inside_file=input("String contained in the file (e.g. WGS84):  \n \n")
string_inside_file=".sgy"
print('The extension is {}'.format(string_inside_file))
filtered_list=filter_list(file_list,string_inside_file)
print("\n")
print("\n")
print(" SEGY files to count traces:")
print("\n")
print(filtered_list)
print("\n")
print("Number of files found: {} Files ".format(len(filtered_list)))

pause=input("Press Enter to continue ...")
traces=[]
samples=[]
length=[]
DT=[]
for file in filtered_list:
    print("\n")
    print("Reading file: {}".format(file))
    n_traces, sample_rate, n_samples, record_length,twt=SEGY_extract_sbp(file)
    print("Data read")
    traces.append(n_traces)
    samples.append(n_samples)
    length.append(record_length)
    DT.append(sample_rate)
    #pause=input("Press Enter to continue ...")
    print("Next file ...")

#print(filtered_list)
#print(TRACES_LIST)
#dest_file=r"D:\SEGY\SEGY_Files\sbp\SUPERB_modified\Processed_data_corrected\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_A\Raw_Data_no_corrections_applied\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\Raw_Data_no_corrections_applied\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_C\Raw_Data_no_corrections_applied\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_C\Processed_data_Heave_Corrected_depth_corrected_as_appropriate\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\Raw_Data_no_corrections_applied\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\Processed_data_Heave_Corrected_depth_corrected_as_appropriate\traces.txt"
#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_B\1.0_RAW_SMT_XY_ICHB\traces.txt"

#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\1.0_PROCESSED_SMT_XY\traces.txt"


#dest_file=r"D:\SEGY\SEGY_Files\sbp\ICHALKIL_SUPERB\1.0_RAW_SMT_XY_ICHSB\traces.txt"

#dest_file=r'D:\SEGY\SEGY_Files\sbp\ICHALKIL_SB_TO_ICKALKIL_A\1.0_PROCESSED-SMT-XY\3.6_Block-6\traces.txt'
dest_file=PATH_IN+'\\traces2.txt'



with open(dest_file, 'w') as f:
    for file,traces,length,samples,dt in zip(filtered_list,traces,length,samples,DT):
        #f.write(archive+'\t\t\t\t'+shift+'\n')
        #f.write('{}\t{}\t{:.2f}\t{}\t{:.2f}\n'.format(str(file[-20:-4].strip()), str(traces),length,samples,dt))
        f.write('{}\t{}\t{:.2f}\t{}\t{:.2f}\n'.format(str(file[int(len_path_in)+1:-4].strip()), str(traces),length,samples,dt))
        #print(archive)
    
    print("File written successfully")
    
    f.close()

print("\n")
print('#                                                                                                      #')
print('#                           DONE:    :) :) NOT SMILING?   FUCK U THEN            #')  
print("\n")   