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

print("\n Busqueda con multindex .......")
#Toda las venta en sucurcal 'Sur'
print(df_multi_index.loc['Sur'])
print("#########################")
# #ventas centro el '2024-06-01'
# print(df_multi_index.loc[('Sur','2024-01-01')])
# print("---------%")
#ventas del categoria Electrónica en el Sur y '2024-06-01'
print(df_multi_index.loc['Sur','2024-01-01','Electrónica'])
# ventas_elec = df_multi_index.xs('Electronica',level="Categoria")
# # Resumen diario de todas las sucursusal solo electronica
# print("\n -------Resumen diario------")-----
# resumen_elec = (
#     ventas_elec.assign.lambda...
# )






#Paso2 : Agrupamiento y agregacion
#Agrupar por sucursal y categoria y luego vamos hacer operaciones
df_grouped= df_ventas.groupby(['Sucursal','Categoria'])['Ventas'].sum().reset_index()
print("\n ######## Paso2. agrupacion y agregacion ########")
print(df_grouped.head(10))
print("\n ######## Paso2. Ejemplo2 agrupacion y agregacion ########")
df_grouped= df_ventas.groupby(['Sucursal','Categoria'])[['Ventas','Unidades']].agg(['sum','mean',"max","min"]).reset_index()
print(df_grouped.head(10))

#Paso3 . Transformacion  de estructura con Pivot
#Agrupamos primero por fecha y categoria para evitar duplicados
print("----------Paso3. Pivot Table----")
df_pivotable = df_ventas.groupby(["Fecha","Categoria"])['Ventas'].sum().reset_index()
print(df_pivotable.head(10))
#Transformar el dataframe a formato ancho(Categoria como columnas)
df_pivot = df_pivotable.pivot(index='Fecha', columns='Categoria', values='Ventas')
print("######3.Tabla pivoteada######")
print(df_pivot.head(10))


