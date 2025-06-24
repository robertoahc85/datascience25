import pandas as pd

df_ventas = pd.read_csv("entradas/ventas_diarias.csv")
df_empleados = pd.read_csv("entradas/empleados_sucursal.csv")
df_productos = pd.read_csv("entradas/productos_categoria.csv")

#======Vista Previa Dataframe Original=======
print("\n ==== Data Frame Original ====")
print(df_ventas.head(9))

#Paso1: Indexacion Jerárquica(Multindex)
#Convertimo tres columnas claves en indice jeraquico - Sucusal , Fecha y Categoria
df_multi_index = df_ventas.set_index(['Sucursal','Fecha', 'Categoria'])
print("\n ------------Paso1. Multidex Aplicado ----------")
print(df_multi_index.head(10))
