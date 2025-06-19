import pandas as pd

# Crear DataFrames de ejemplo
data_referencia = {'id': [1, 2, 3, 4], 'nombre': ['A', 'B', 'C', 'D']}
df_referencia = pd.DataFrame(data_referencia)

data_destino = {'valor_a_mapear': ['1', '3', '2', '4']}
df_destino = pd.DataFrame(data_destino)

# Crear el diccionario de mapeo
diccionario_mapeo = dict(zip(df_referencia['id'], df_referencia['nombre']))

# Mapear los valores
df_destino['nombre_mapeado'] = df_destino['valor_a_mapear'].map(diccionario_mapeo)

# Imprimir el resultado
print(df_destino)