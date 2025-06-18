import pandas as pd
# Cargar el archivo excel
df = pd.read_excel('entradas/reporte.xlsx', sheet_name='Ventas2023')
print(df.head())