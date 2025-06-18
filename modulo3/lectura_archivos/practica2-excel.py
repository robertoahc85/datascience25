import pandas as pd
# Cargar el archivo excel
# df = pd.read_excel('entradas/reporte.xlsx',sheet_name="productos2023")
# print(df)
# df2 = pd.read_excel('entradas/reporte.xlsx',sheet_name="Ventas2023")
# print(df2)
# data = {"Producto": ["Camiseta", "Pantalón", "Zapatos"],
#         "Precio": [20, 30, 50],}
# df = pd.DataFrame(data)
# df.to_excel('salidas/productos2024.xlsx', index=False, sheet_name='Resultados')
# data = {"Nombre": ["Roberto", None], "edad": [25, 30]}
data = pd.read_excel('entradas/personas.xlsx')
df =pd.DataFrame(data)
df.to_excel('salidas/personas2.xlsx', index=False, sheet_name='Datos', na_rep="Sin datos")
