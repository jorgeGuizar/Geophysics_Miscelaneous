import pandas as pd
import numpy as np
import os

def split_excel_sheets(file_path):
    """
    Reads an Excel file and splits it into multiple sheets based on unique values in a specified column.

    Args:
        file_path (str): The path to the Excel file.
    """
    try:
        # Read all sheets from the Excel file
        xls = pd.ExcelFile(file_path)
        sheet_names = xls.sheet_names

        # Iterate through each sheet
        for sheet_name in sheet_names:
            df = pd.read_excel(xls, sheet_name)
            print()

            # Check if the DataFrame is empty
            if df.empty:
                print(f"Warning: Sheet '{sheet_name}' is empty and will be skipped.")
                continue

            # Get unique values from the first column (you can change the column index as needed)
            valores_a=np.arange(0,47.501,.5)
            valores_b=valores_a+.500
            #unique_values = df.iloc[:, 0].unique()

            # Create a new workbook to store the split sheets
           #writer = pd.ExcelWriter(f"{sheet_name}_split_2.xlsx", engine='xlsxwriter')
            writer = pd.ExcelWriter(f"REGISTRO DE PROFUNDIDADES OGD20ØX47.5KM_VFP PLTM GEN TENTOK-A_PLTM EXIST KAB-C_split.xlsx", engine='xlsxwriter')
            

             # Iterate through each unique value and create a new sheet
            for val_inf,val_sup in zip(valores_a,valores_b):
                # Filter the DataFrame for the current value
                #filtered_df = df[df.iloc[:, 0] == value]
                filtered_df=df[(df['KP\n[km]'] >= val_inf) & (df['KP\n[km]'] <= val_sup )].copy()
                # Write the filtered DataFrame to a new sheet
                sheet_f=f'{str(val_inf)}-{str(val_sup)}' 
                filtered_df.to_excel(writer, sheet_name=str(sheet_f), index=False)
            writer.close()
            print(f"Sheets from '{sheet_name}' split successfully and saved to '{file_path[:-5]}_split_2.xlsx'")
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    file_path = r'C:\Users\gener\Documents\Permaducto\Tentok-A\REGISTRO DE PROFUNDIDADES OGD20ØX47.5KM_VFP PLTM GEN TENTOK-A_PLTM EXIST KAB-C.xlsx'#"  # Replace with your file path
    split_excel_sheets(file_path)