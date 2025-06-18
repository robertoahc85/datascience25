import pandas as pd
# Cargar el archivo CSV
# df = pd.read_csv('entradas/estudiantes.csv',sep=";")
# print(df)
# Eliminar filas valores nulos
df =pd.read_csv('entradas/ventas.csv', na_values=["N/A"])
df = df.dropna()  # Eliminar filas donde 'Precio' es NaN
# print(df)
#escribir un archivo CSV
data= {"Producto": ["Peras", "Durazno"],"Precio": [0.5, 0.3]}
df = pd.DataFrame(data)
df.to_csv('entradas/productos.csv', index=False)
