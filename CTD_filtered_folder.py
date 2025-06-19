import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from scipy.signal import savgol_filter
from scipy.interpolate import splrep, BSpline
#########################################################################################
# FUNCTIONS DEFINITIONS
#########################################################################################
def find_folders(path):
  """
    Finds all folders within a given directory path.

    Args:
        path: The path to the directory to search.

    Returns:
        A list of folder names found in the directory.
  """
  folders = []
  for item in os.listdir(path):
    item_path = os.path.join(path, item)
    if os.path.isdir(item_path):
      folders.append(item)
  return folders

def iterate_folders(path,lista_folders):
    folder_path=[]
    for folder in lista_folders:
        fpath = os.path.join(path, folder)
        folder_path.append(fpath)
        # Check if the item is a directory
        if os.path.isdir(fpath):
            print(f"Folder: {fpath}")
            # You can add more processing logic here if needed
            # For example, you could read files in the folder or perform other operations
            # files = os.listdir(folder_path)
            # for file in files:
            #     print(f"  File: {file}")
        else:
            print(f"{fpath} is not a folder.")
            
    return folder_path

def create_folder_output(path_list):
    out=r'out_folder'
    for path in path_list:
        # Check if the directory already exists
        isExist = os.path.exists(os.path.join(path, out))
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(os.path.join(path, out))
            print("The new directory is created!: {}".format(os.path.join(path, out)))
        else:
            print("The directory already exists!: {}".format(os.path.join(path, out)))
    
    
    #isExist = os.path.exists(path)
    #if not isExist:
    # Create a new directory because it does not exist
    #    os.makedirs(path)
    #    print("The new directory is created!: {}".format(path))
    
def list_first_level_files(folder_path):
    """
    Lists the first-level files in a folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        list: A list of file names in the folder.
              Returns an empty list if the folder does not exist or if an error occurs.
    """
    try:
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except FileNotFoundError:
        print(f"Error: Folder '{folder_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def increasing_values_filter(df):
    increasing_values = [df['Depth(m)'].iloc[0]]
    print(increasing_values)
    # Iterate through the values in the 'Depth(m)' column
    for val in df['Depth(m)'].iloc[1:]:
        if val > increasing_values[-1]:
            increasing_values.append(val)
    df_2=df[df['Depth(m)'].isin(increasing_values)].copy()
    return df_2

def decreasing_values_filter(df):
    max_idx = df['Depth(m)'].idxmax()
    #print(max_idx,df_final_1['Depth(m)'].iloc[max_idx])
    start_value = df.loc[max_idx, 'Depth(m)']
    #print(start_value)
    #input("Press Enter to continue...")
    decreasing_values = [start_value]
    #print(decreasing_values)
    indices = [max_idx]
    # Iterate through the values in the 'Depth(m)' column starting from the max value
    for i in range(max_idx + 1, len(df)):
        current_value = df.loc[i, 'Depth(m)']
        print(current_value)
        if current_value < decreasing_values[-1]:
            decreasing_values.append(current_value)
            indices.append(i)
    df_2=df[df['Depth(m)'].isin(decreasing_values)].copy()  
    return df_2

def getExponential(input_data, alpha):
    """
    Smoothing is a method to filter time series. 

    Args:
        input_data (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    yh=np.zeros(len(input_data))
    yh[0]=input_data[0]
    for i in range(1, len(input_data)):
        yh[i] = alpha * input_data[i] + (1 - alpha) * yh[i - 1]
    return yh

def getSavitzkyGolay(input_data, window_length,polyorder):
    """
    Smoothing is a method to filter time series. 

    Args:
        input_data (_type_): _description_
        alpha (float, optional): _description_. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    #polyorder=3
    yh = savgol_filter(input_data, window_length, polyorder)
    return yh

def getConvolve(input_data,box_pts):
    """_summary_

    Args:
        input_data (_type_): _description_
        box_pts (_type_): _description_

    Returns:
        _type_: _description_
    """
    box = np.ones(box_pts)/box_pts
    yh = np.convolve(input_data, box, mode='same')
    return yh



    

def smooth_data(method, input_data, **kwargs):
    """
    Applies a smoothing method to input data using the specified method.

    Args:
        method (str): The name of the smoothing method ('exponential', 'savitzky_golay', or 'convolve').
        input_data (array-like): The input time series data.
        **kwargs: Additional keyword arguments required by the selected smoothing function.

    Returns:
        numpy.ndarray: Smoothed time series.

    Raises:
        ValueError: If the specified method is not supported.
    """
    smoothing_methods = {
        'exponential': getExponential,
        'savitzky_golay': getSavitzkyGolay,
        'convolve': getConvolve
    }

    if method not in smoothing_methods:
        raise ValueError(f"Unsupported method: {method}. Choose from {list(smoothing_methods.keys())}")

    return smoothing_methods[method](input_data, **kwargs)




def plot_Depth_vs_Parameters(Dataframe,folder_images,smoothtype):
    """_summary_

    Args:
        Dataframe (_type_): _description_
        folder_images (_type_): _description_
        smoothtype (_type_): Smooth type 
    """
    depth=Dataframe['Depth(m)'].values
    pressure=Dataframe['Pressure (DBar)'].values
    temp=Dataframe['Temperature (DegC)'].values
    conductivity=Dataframe['Conductivity (mS/cm)'].values
    salinity=Dataframe['Salinity (PSU)'].values
    velocity=Dataframe['Sound Velocity (MS-1)'].values
    density=Dataframe['Density (kg/M3)'].values
    #input_data=velocity
    
    if smoothtype=='Exponential':
        vel_smooth = smooth_data('exponential', velocity, alpha=0.6)  
    elif smoothtype=='Convolution':
        vel_smooth = smooth_data('convolve', velocity, box_pts=3)
    elif smoothtype=='SavitzkyGolay':
        vel_smooth=smooth_data('savitzky_golay', velocity, window_length=5, polyorder=3)
    else:
        vel_smooth=velocity.copy()
        
        
    #vel_smooth= {'Exponential':getExponential, 'SavitzkyGolay':getSavitzkyGolay, 'Convolution':getConvolve}[smoothtype](input_data,twav)#getExponentialSmooth(velocity, alpha=.8)
    # Create a figureS and axis
    ##################################################################################
    ##################################################################################
    # plot 1
    data=[pressure,temp,conductivity,salinity,velocity,vel_smooth,density]
    labels=['Presión(DBar)', 'Temperatura(°C)', 'Conductividad(mS/cm)', 'Salinidad(PSU)', 'Velocidad del Sonido(m/s)', 'Velocidad del Sonido(m/s)','Densidad(kg/m3)']
    labels_text=['Presion', 'Temperatura', 'Conductividad', 'Salinidad', 'VelocidadSonido', 'VelocidadSonidoSmooth','Densidad']
    i=0
    archivo=file[:-4]
    for etiq in labels:
        fig, ax1 = plt.subplots(figsize=(14,12),dpi=100)
        # Plot the data
        ax1.plot(data[i],depth, label=etiq, color='blue',linestyle='solid',linewidth=1.5)
        # Set labels and title
        ax1.set_title('{}\n{}\n'.format(archivo,etiq),fontsize=14,weight='bold')
        ax1.set_ylabel('Profundidad (m)',fontsize=11)
        promedio= np.mean(data[i])
        ax1.set_xlabel(etiq +'\n',fontsize=11)
        ax1.xaxis.tick_top()
        ax1.set_xlim([np.min(data[i]),np.max(data[i])+1])
        ax1.set_ylim([0,np.max(depth)])
        ax1.yaxis.set_inverted(True)  # inverted axis with autoscaling
        ax1.xaxis.set_label_position('top') 
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),fancybox=True, shadow=True, ncol=2, fontsize=13)
        figtext_args = (0.5, .1, 'Promedio {}: {}'.format(etiq,round(promedio,2)))#, {'ha': 'center', 'va': 'bottom', 'fontsize': 12})
        figtext_kwargs=dict(horizontalalignment ="center", fontsize = 14, color ="black",  style ="oblique", wrap = True,bbox ={'facecolor':'grey',  
                   'alpha':0.3, 'pad':5}) 
        text = fig.text(*figtext_args, **figtext_kwargs)#(0.50, 0.01, '', horizontalalignment='center', wrap=True, bbox ={'facecolor':'grey',  'alpha':0.3, 'pad':5} )
        ax1.tick_params(axis='both', which='major', labelsize=11)
        #ax1.set_xlim([np.min(data[i]),np.max(data[i])])
        #ax1.set_ylim([np.min(depth),np.max(depth)])
        #i+=1
        # Add a legend
        #ax1.legend()
        plt.grid(True)
        #fig.tight_layout(rect=(0,.05,1,1)) 
        # Show the plot
        #plt.show()
        #fig.savefig(os.path.join(folder_images,'{}_figura_{}.png'.format(archivo,etiq)), dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(folder_images,f'{archivo}_figura_{labels_text[i]}.png'), dpi=fig.dpi, bbox_inches='tight')
        i+=1
        #plt.show()
        #plt.savefig(os.path.join(folder,'{}_{}.png'.format(file[:-4],etiq)))#, dpi=200, bbox_inches='tight')
    
    #plt.close() 
    
def plot_Depth_vs_Velocity_ALL(Dataframe,folder_images,smoothtype):
    """_summary_

    Args:
        Dataframe (_type_): _description_
        folder_images (_type_): _description_
        smoothtype (_type_): Smooth type 
    """
    
    depth=Dataframe['Depth(m)'].values
    pressure=Dataframe['Pressure (DBar)'].values
    temp=Dataframe['Temperature (DegC)'].values
    conductivity=Dataframe['Conductivity (mS/cm)'].values
    salinity=Dataframe['Salinity (PSU)'].values
    velocity=Dataframe['Sound Velocity (MS-1)'].values
    density=Dataframe['Density (kg/M3)'].values
    #input_data=velocity
    
    if smoothtype=='Exponential':
        vel_smooth = smooth_data('exponential', velocity, alpha=0.6)  
    elif smoothtype=='Convolution':
        vel_smooth = smooth_data('convolve', velocity, box_pts=3)
    else:
        vel_smooth=smooth_data('savitzky_golay', velocity, window_length=5, polyorder=3)
        
        
    #vel_smooth= {'Exponential':getExponential, 'SavitzkyGolay':getSavitzkyGolay, 'Convolution':getConvolve}[smoothtype](input_data,twav)#getExponentialSmooth(velocity, alpha=.8)
    # Create a figureS and axis
    ##################################################################################
    ##################################################################################
    # plot 1
    data=[pressure,temp,conductivity,salinity,velocity,vel_smooth,density]
    labels=['Presión(DBar)', 'Temperatura(°C)', 'Conductividad(mS/cm)', 'Salinidad(PSU)', 'Velocidad del Sonido(m/s)', 'Velocidad del Sonido(m/s)','Densidad(kg/m3)']
    labels_text=['Presion', 'Temperatura', 'Conductividad', 'Salinidad', 'VelocidadSonido', 'VelocidadSonidoSmooth','Densidad']
    i=0
    archivo=file[:-4]
    for etiq in labels:
        fig, ax1 = plt.subplots(figsize=(14,12),dpi=100)
        # Plot the data
        ax1.plot(data[i],depth, label=etiq, color='blue',linestyle='solid',linewidth=1.5)
        # Set labels and title
        ax1.set_title('{}\n{}\n'.format(archivo,etiq),fontsize=14,weight='bold')
        ax1.set_ylabel('Profundidad (m)',fontsize=11)
        promedio= np.mean(data[i])
        ax1.set_xlabel(etiq +'\n',fontsize=11)
        ax1.xaxis.tick_top()
        ax1.set_xlim([np.min(data[i]),np.max(data[i])+1])
        ax1.set_ylim([0,np.max(depth)])
        ax1.yaxis.set_inverted(True)  # inverted axis with autoscaling
        ax1.xaxis.set_label_position('top') 
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),fancybox=True, shadow=True, ncol=2, fontsize=13)
        figtext_args = (0.5, .1, 'Promedio {}: {}'.format(etiq,round(promedio,2)))#, {'ha': 'center', 'va': 'bottom', 'fontsize': 12})
        figtext_kwargs=dict(horizontalalignment ="center", fontsize = 14, color ="black",  style ="oblique", wrap = True,bbox ={'facecolor':'grey',  
                   'alpha':0.3, 'pad':5}) 
        text = fig.text(*figtext_args, **figtext_kwargs)#(0.50, 0.01, '', horizontalalignment='center', wrap=True, bbox ={'facecolor':'grey',  'alpha':0.3, 'pad':5} )
        ax1.tick_params(axis='both', which='major', labelsize=11)
        #ax1.set_xlim([np.min(data[i]),np.max(data[i])])
        #ax1.set_ylim([np.min(depth),np.max(depth)])
        #i+=1
        # Add a legend
        #ax1.legend()
        plt.grid(True)
        #fig.tight_layout(rect=(0,.05,1,1)) 
        # Show the plot
        #plt.show()
        #fig.savefig(os.path.join(folder_images,'{}_figura_{}.png'.format(archivo,etiq)), dpi=200, bbox_inches='tight')
        fig.savefig(os.path.join(folder_images,f'{archivo}_figura_{labels_text[i]}.png'), dpi=fig.dpi, bbox_inches='tight')
        i+=1

def create_output_folder(folderpath):
    
    isExist = os.path.exists(folderpath)
    #print("The directory exists!: {}".format(folderpath))
    if not isExist:
        os.makedirs(folderpath)#Create a new directory because it does not exist
        print("The new directory is created!: {}".format(folderpath))
    
    print("The directory exists!: {}".format(folderpath))

def Dataframe_cleaning(Dataframe):
    """_summary_

    Args:
        Dataframe (_type_): _description_

    Returns:
        _type_: _description_
    """
    Dataframe.rename(columns={"Unnamed: 1": "Depth(m)",'Depth(m)':  'Pressure (DBar)', \
        'Pressure (DBar)':'Temperature (DegC)','Temperature (DegC)':'Conductivity (mS/cm)','Conductivity (mS/cm)':'Salinity (PSU)', \
        'Salinity (PSU)':'Sound Velocity (MS-1)','Sound Velocity (MS-1)':'Density (kg/M3)','Density (kg/M3)':'vacio'}, inplace=True)
    Dataframe.drop(['vacio'], axis=1, inplace=True)
    Dataframe.drop(['Date/Time'], axis=1, inplace=True)
    columns_to_convert = ['Depth(m)','Pressure (DBar)','Temperature (DegC)','Conductivity (mS/cm)','Salinity (PSU)',\
        'Sound Velocity (MS-1)','Density (kg/M3)']
    Dataframe[columns_to_convert] = Dataframe[columns_to_convert].astype(float)
    Dataframe_final_1=Dataframe[(df['Pressure (DBar)'] > 0) & (Dataframe['Conductivity (mS/cm)'] > 0) & (Dataframe['Depth(m)'] > 0) \
        & (Dataframe['Sound Velocity (MS-1)']>1400) & (Dataframe['Temperature (DegC)']>20) & (Dataframe['Salinity (PSU)']>0) ].copy()
    #reset index
    Dataframe_final_1=Dataframe_final_1.reset_index(drop=True)
        #print(df_final_1)
        #calcuklate increasing and decreasing values
    Dataframe_final_1_dec=decreasing_values_filter(Dataframe_final_1)
    Dataframe_final_1_inc=increasing_values_filter(Dataframe_final_1)
    #drop duplicates
    Dataframe_unique_dec= Dataframe_final_1_dec.drop_duplicates(subset=['Depth(m)'], keep='first')
    Dataframe_unique_inc= Dataframe_final_1_inc.drop_duplicates(subset=['Depth(m)'], keep='first')
    #cleaning outliers
    
    
    
    return Dataframe_unique_dec, Dataframe_unique_inc


def filter_outliers(df, column):
    """Filters outliers from a pandas DataFrame column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to filter.

    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """
    lower_limit = df[column].quantile(0.05)
    upper_limit = df[column].quantile(0.95)
    filtered_df = df[(df[column] >= lower_limit) & (df[column] <= upper_limit)]
    return filtered_df
#########################################################################################
# END FUNCTIONS DEFINITIONS END
#########################################################################################
#import sys
#sys.modules[__name__].__dict__.clear()
path=r'C:\Users\gener\Documents\Permaducto\CTD'
images_path=r'C:\Users\gener\Documents\Permaducto\CTD_IMAGES'
# isExist = os.path.exists(path)
# if not isExist:
#    # Create a new directory because it does not exist
#    os.makedirs(path)
#    print("The new directory is created!: {}".format(path))
   
create_output_folder(path)

create_output_folder(images_path)
   
# print lista de folders
lista_folders=find_folders(path)
# create the path for folders inside the main folder
# here we will read the files
folder_paths=iterate_folders(path,lista_folders)
print("folders son")
print(folder_paths)
# create the output folder inside one oof the folders
#print of the out folder
#create_folder_output(folder_paths)
#create_folder_output(images_path)
# list the files inside the folders

for folder in folder_paths:
    print('forder:{}'.format(folder))
    files_list_per_folder = list_first_level_files(folder)
    print(f"Files in {folder}: {files_list_per_folder}")
    for file in files_list_per_folder:
        df=pd.read_csv(os.path.join(folder, file), header=0 )

        df_unique_inc, df_unique_dec=Dataframe_cleaning(df)
        # outliers filter 5-95%
        
        #plot the paramaters valores que se incrementan en profundidad
        plot_Depth_vs_Parameters(df_unique_inc,images_path,smoothtype='Exponential')
        
        
        
        df_final_concat = pd.concat([df_unique_inc, df_unique_dec], axis=0)  
        df_final_concat =df_final_concat.reset_index(drop=True)
        final_file=os.path.join(folder, 'out_folder', file)
        print(folder)
        print(file)
        print(df_final_concat)
        df_final_concat.to_csv(final_file, index=False)
        # csv(output_file, index=False)

