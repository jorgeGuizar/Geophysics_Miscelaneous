import numpy as np
import matplotlib.pyplot as plt

input_data=input("Enter the  type of data e.g. 1: shot ; 2: gathers ; 3: stack     .......     ")

if input_data == '1':
    print("You have selected shot data")
elif input_data == '2':
    print("You have selected gathers data")
else:
    print("You have selected stack data")
    
Size_text=320 #bytes
Size_binary=400 #bytes
 




TOTAL_size=Size_text+Size_binary