import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.ticker import FormatStrFormatter
import tkinter
import random
#import sys
#sys.modules[__name__].__dict__.clear()
#path=r'D:\demar_3.nmp\demar_3\images_GASODUCTO_BN_16Ø _280425'
# isExist = os.path.exists(path)
# if not isExist:
#    os.makedirs(path)#Create a new directory because it does not exist

#    print("The new directory is created!: {}".format(path))

def create_output_folder(folderpath):
    
    isExist = os.path.exists(folderpath)
    if not isExist:
        os.makedirs(folderpath)#Create a new directory because it does not exist

    print("The new directory is created!: {}".format(folderpath))
    

def get_sheet_filter(xls_file_path, symbol):
    """
    Returns a function that filters sheet names in a xls based on the presence of ' any simbol or string'.
    """
    xl_2 = pd.ExcelFile(xls_file_path)
    sheet_list_raw=list(xl_2.sheet_names)
    sheets_list_filtered=list(filter(lambda k: symbol in k, xl_2.sheet_names))
    return sheet_list_raw,sheets_list_filtered

def add_sheet_to_filtered(lista_filtered, lista_add):
    """
    Adds the filtered sheets to the raw list.
    """
    for sheet in lista_add:
        if sheet not in lista_filtered:
            lista_filtered.append(sheet)
    return lista_filtered

def extract_and_concat_excel(file_path,lista_filtered):
    """
    Extracts all sheets from an Excel file and concatenates them into a single DataFrame.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pandas.DataFrame: A DataFrame containing the combined data from all sheets,
                         or None if an error occurs.
    """
    try:
        excel_data = pd.read_excel(file_path, sheet_name=lista_filtered, usecols='A:G', skiprows=6, header=0, index_col=None, engine='openpyxl')

        if not excel_data:
            print("No sheets found in the Excel file.")
            return None

        all_sheets_df = pd.concat(excel_data.values(), ignore_index=False)
        #copy the dataframe"
        df_1=all_sheets_df.copy()
        #drop Nan values"
        df_2 = df_1.dropna()
        # reset the index"
        df_3=df_2.reset_index(drop=True)
        # filter the dataframe with KP>=0
        df_final=df_3[df_3['KP (km)'] >= 0].copy()
        return df_final

    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
        return None
    except Exception as e:
         print(f"An error occurred: {e}")
         return None

def Dataframe_Values_to_list(dataframe,lim_min_KP,lim_max_KP):
    """
    Convert DataFrame values  to vectors  and lists
    """
    codigos_eventos=['JP-','MC-','CS-','CSS']
    df4=dataframe[(dataframe['KP (km)'] >= lim_min_KP) & (dataframe['KP (km)'] <= lim_max_KP )].copy()
    df1 = df4[df4['Cód.'].str.contains('JP-|MC-|CS-|CSS')].copy()
    KP=np.array(df1['KP (km)'].values)
    Z=np.array(df1['Prof. (m)'].values)
    Codigo=list(df1['Cód.'].values)
    return  KP,Z,Codigo
#titulo=r'GASODUCTO BN DE 16Ø X 0.5 KM DE INTERCONEXIÓN SUBMARINA CON GSD BN DE 16Ø YAXCHE-A/TLACAME-A HACIA MULACH-A'
def plot_xls_events(KP,LT,FT,ENT,ENT_PROYECTO,NMLM,COBERTURA,NMC,KP_evento,depth_evento,Codigo_evento,tramo,titulo,folderpath,fig_1=1,fig_2=2,fig_3=3):
    cm = 1/2.54   

    maxLT=np.max(LT)
    maxFT=np.max(FT)
    maxNMLM=np.max(NMLM)
    
    minLT=np.min(LT)
    minFT=np.min(FT)
    minNMLM=np.min(NMLM)
    
    
    max_total=np.max([maxLT,maxFT,maxNMLM])
    min_total=np.min([minLT,minFT,minNMLM])
    ##################################################################################
    ##################################################################################
    # plot 1
    fig, ax1 = plt.subplots(figsize=(25*cm,14*cm))
    ax2 = ax1.twinx()
    ax1.plot(KP,LT, label='Lomo de Tubo (LT)',color='red', linestyle='dotted')
    ax1.plot(KP,FT, label='Fondo del Tubo (FT)', color='red', linestyle='dashed')
    ax1.plot(KP,NMLM, label='Nivel Medio del Lecho Marino (NMLM)', color='blue', linestyle='solid')
    ax2.plot(KP,ENT, label='Enterramiento (Ent.)',color='brown')
    ax2.plot(KP,ENT_PROYECTO, label='Enterramiento de Proyecto', color='green')
    #ax1.plot(KP,NMC, label='Nivel Medio del Canal (NMC)', color='orange', linestyle='-.')
    ax1.set_title(titulo +'\n KP-{} \n'.format(tramo), fontsize=20, weight='bold')
    ax1.set_xlabel('KP [km]',fontsize=15)
    ax1.set_ylabel('Profundidad [m]',fontsize=15)
    ax2.set_ylabel('Enterramiento de Proyecto [m]',fontsize=15)
    ax1.grid(True)
    ax1.xaxis.set_label_position('top') 
    ax1.xaxis.tick_top()
    ax2.set_ylim([-3, 3])
    ax1.set_ylim([min_total-1, max_total+1])
    box1 = ax1.get_position()
    box2 = ax2.get_position()
    ax1.set_position([box1.x0, box1.y0 + box1.height * 0.05,
                    box1.width, box1.height * .9])
    ax2.set_position([box2.x0, box2.y0 + box2.height * 0.05,
                    box2.width, box2.height * .9])

    ax1.legend(loc='upper right', bbox_to_anchor=(0.5, -0.02),#.1
            fancybox=True, shadow=True, ncol=3,fontsize=15)
    ax2.legend(loc='upper left', bbox_to_anchor=(0.5, -0.02),
            fancybox=True, shadow=True, ncol=3,fontsize=15)
    ax1.set_yticks(np.arange(min_total-1, max_total+1, step=.5))
    ax1.yaxis.set_inverted(True) 
    ax2.yaxis.set_inverted(True) 
    ax1.set_xlim([np.min(KP),np.max(KP)])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig = plt.gcf()
    plt.show()
    fig.savefig(os.path.join(folderpath,'{}_figura_{}.png'.format(tramo,fig_1)), dpi=200, bbox_inches='tight')
    min_LT=np.min(LT)
    min_FT=np.min(FT)
    min_NMLM=np.min(NMLM)
    min_COBERTURA=np.min(COBERTURA)
    min_NMC=np.min(NMC)
    
    max_LT=np.max(LT)
    max_FT=np.max(FT)
    max_NMLM=np.max(NMLM)
    max_COBERTURA=np.max(COBERTURA)
    max_NMC=np.max(NMC)
    max_total_2=np.max([max_LT,max_FT,max_NMLM,max_COBERTURA, max_NMC])
    min_total_2=np.min([min_LT,min_FT,min_NMLM,min_COBERTURA, min_NMC])


    # plot 2
    fig, ax1 = plt.subplots(figsize=(27*cm,14*cm))
    ax1.plot(KP,LT, label='Lomo de Tubo (LT)',color='red', linestyle='dotted')
    ax1.plot(KP,FT, label='Fondo del Tubo (FT)', color='red', linestyle='dashed')
    ax1.plot(KP,NMLM, label='Nivel Medio del Lecho Marino (NMLM)')
    ax1.plot(KP,COBERTURA, label='Cobertura',color='lightgreen')
    ax1.plot(KP,NMC, label='Nivel Medio del Canal (NMC)', color='orange', linestyle='-.')
    i=0
    for evento in Codigo_evento:
        random_float_range = np.round(random.uniform(-.5, -1.1),2)
        ax1.annotate(evento, xy=(KP_evento[i],depth_evento[i]), xytext=(KP_evento[i],depth_evento[i]+random_float_range), rotation=90,ha='center',
                     va='center',arrowprops=dict(facecolor='black', shrink=0.05, width=.5),fontsize=12)
        i+=1
    ax1.set_title(titulo +'\n KP-{} \n'.format(tramo),fontsize=20,weight='bold')
    ax1.set_xlabel('KP [km]',fontsize=15)
    ax1.set_ylabel('Profundidad [m]',fontsize=15)
    ax1.grid(True)
    ax1.xaxis.set_label_position('top') 
    ax1.xaxis.tick_top()
    ax1.set_ylim([min_total_2-1.5, max_total_2+1.5])
    ax1.set_xlim([np.min(KP),np.max(KP)])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
            fancybox=True, shadow=True, ncol=5, fontsize=15)
    ax1.yaxis.set_inverted(True)  
    ax1.set_yticks(np.arange(min_total_2-1.5, max_total_2+1.5, step=.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig = plt.gcf()
    plt.show()
    fig.savefig(os.path.join(folderpath,'{}_figura_{}.png'.format(tramo,fig_3)), dpi=200,bbox_inches='tight')
    
    
    # plot 3
    fig, ax1 = plt.subplots(figsize=(27*cm,14*cm))
    ax1.plot(KP,LT, label='Lomo de Tubo (LT)',color='red', linestyle='dotted')
    ax1.plot(KP,FT, label='Fondo del Tubo (FT)', color='red', linestyle='dashed')
    ax1.plot(KP,NMLM, label='Nivel Medio del Lecho Marino (NMLM)')
    ax1.plot(KP,COBERTURA, label='Cobertura',color='lightgreen')
    #ax1.plot(KP_evento,depth_evento,'ko', label='Eventos', markersize=10)
    i=0
    for evento in Codigo_evento:
        random_float_range = np.round(random.uniform(-.5, -1.1),2)
        ax1.annotate(evento, xy=(KP_evento[i],depth_evento[i]), xytext=(KP_evento[i],depth_evento[i]+random_float_range), rotation=90,ha='center',
                     va='center',arrowprops=dict(facecolor='black', shrink=0.05, width=.5),fontsize=12)
        i+=1
    ax1.set_title(titulo +'\n KP-{} \n'.format(tramo),fontsize=20,weight='bold')
    ax1.set_xlabel('KP [km]',fontsize=15)
    ax1.set_ylabel('Profundidad [m]',fontsize=15)
    ax1.grid(True)
    ax1.xaxis.set_label_position('top') 
    ax1.xaxis.tick_top()
    ax1.set_ylim([min_total_2-1.5, max_total_2+1.5])
    ax1.set_xlim([np.min(KP),np.max(KP)])
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                    box.width, box.height * 0.9])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.02),
            fancybox=True, shadow=True, ncol=4, fontsize=15)
    ax1.yaxis.set_inverted(True)  
    ax1.set_yticks(np.arange(min_total_2-1.5, max_total_2+1.5, step=.5))
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fig = plt.gcf()
    plt.show()
    fig.savefig(os.path.join(folderpath,'{}_figura_{}.png'.format(tramo,fig_2)), dpi=200,bbox_inches='tight')