import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D

xl = pd.ExcelFile(r'C:\Users\LEGA\Documents\PROFUNDIDADES OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-AXANAB-C.xlsx')
sheets_list=list(filter(lambda k: '+' in k, xl.sheet_names))
print(sheets_list)
input("Press Enter to continue...") # Pause for user input

#xls = pd.read_excel(r'C:\Users\LEGA\Documents\PROFUNDIDADES OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-AXANAB-C.xlsx', sheet_name=['0+000-0+500', \
#    '0+501-1+000', '1+001-1+500', '1+501-2+000', '2+001-2+500', '2+501-3+000', '3+001-3+500', '3+501-4+000', '4+001-4+500', '4+501-5+000', \
#    '5+001-5+500', '5+501-6+000', '6+001-6+500', '6+501-7+000', '7+001-7+500', '7+501-8+000', '8+001-8+500', '8+501-9+000', '9+001-9+500', '9+501-10+000', '10+001-10+213'])
xls = pd.read_excel(r'C:\Users\LEGA\Documents\PROFUNDIDADES OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-AXANAB-C.xlsx', sheet_name=sheets_list)

print(xl.sheet_names)
sheet1_df = xls['0+000-0+500']

print(sheet1_df.head())


KP=sheet1_df['KP\n[km]']
LT=sheet1_df['Lomo de Tubo (LT)\n[m]']
FT=sheet1_df['Fondo del Tubo (FT)\n[m]']
ENT=sheet1_df['Enterramiento (Ent.)\n[m]']
ENT_PROYECTO=sheet1_df['Enterramiento de Proyecto\n[m]']
NMLM=sheet1_df['Nivel Medio del Lecho Marino (NMLM)\n[m]']
COBERTURA=sheet1_df['Cobertura  \n[m]']



fig, ax1 = plt.subplots(figsize=(10,8))
ax2 = ax1.twinx()
ax1.plot(KP,LT, label='Lomo de Tubo (LT)',color='red', linestyle='dotted')
ax1.plot(KP,FT, label='Fondo del Tubo (FT)', color='red', linestyle='dashed')
ax2.plot(KP,ENT, label='Enterramiento (Ent.)',color='brown')
ax2.plot(KP,ENT_PROYECTO, label='Enterramiento de Proyecto', color='green')
ax1.plot(KP,NMLM, label='Nivel Medio del Lecho Marino (NMLM)')
ax1.set_title('OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-A/XANAB-C  \n')
ax1.set_xlabel('KP [km]')
ax1.set_ylabel('Profundidad [m]')
ax2.set_ylabel('Enterramiento de Proyecto [m]')
ax1.grid(True)
ax1.xaxis.set_label_position('top') 
ax1.xaxis.tick_top()
ax2.set_ylim([-3, 3])
ax1.set_ylim([20, 30])
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
ax1.yaxis.set_inverted(True) 
ax2.yaxis.set_inverted(True) 
ax1.set_yticks(np.arange(20, 30, step=.5))
#fig.tight_layout()
plt.show()
fig.savefig('temp.png', dpi=fig.dpi)

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.plot(KP,LT, label='Lomo de Tubo (LT)',color='red', linestyle='dotted')
ax1.plot(KP,FT, label='Fondo del Tubo (FT)', color='red', linestyle='dashed')
ax1.plot(KP,NMLM, label='Nivel Medio del Lecho Marino (NMLM)')
ax1.plot(KP,COBERTURA, label='COBERTURA',color='lightgreen')
ax1.set_title('OGSD 20ØX10.4 KM DE MULACH-A HACIA INTERCONEXIÓN SUBMARINA CON OGD 20Ø TLACAME-A/XANAB-C  \n')
ax1.set_xlabel('KP [km]')
ax1.set_ylabel('Profundidad [m]')
ax1.grid(True)
ax1.xaxis.set_label_position('top') 
ax1.xaxis.tick_top()
ax1.set_ylim([20, 25])
box = ax1.get_position()
ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                 box.width, box.height * 0.9])
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
ax1.yaxis.set_inverted(True)  
ax1.set_yticks(np.arange(20, 25, step=.5))
#fig.tight_layout()
plt.show()
fig.savefig('temp2.png', dpi=fig.dpi)

