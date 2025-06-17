import pandas as pd
# data = {
#     "Nombre": ["Juan", "Ana", "Pedro", "Maria"],
#     "calificaciones": [4, 3, 5, 7],
#     "Asignatura": ["Matematicas", "Ciencias", "Historia", "Lengua"],
# }
# df = pd.DataFrame(data)
# print("DataFrame original:")
# print(df)

# accediendo desde listas
# products = ["Camiseta", "Pantalones", "Zapatos"]
# prices = [20.5, 35.0, 50.0]
# df2 = pd.DataFrame({
#     "Producto": products,
#     "Precio": prices
# })
# print("\nDataFrame desde listas:")
# print(df2)
# acciendo multiples columnas
data2 = {'Nombre': ["Juan", "Ana", "Pedro", "Maria"],
        'Salario': [3000, 4000, 5000, 6000],
        'departamento': ["Ventas", "Marketing", "IT", "Recursos Humanos"]}
df3 = pd.DataFrame(data2)
print("\nDataFrame con múltiples columnas:")
print(df3)
print(df3[['Nombre','Salario']])
print(df3.iloc[3])
print(df3.iloc[1:3])
print(df3[(df3['Salario'] > 4000) & (df3['departamento'] == 'IT')])
print(df3[df3['Nombre'].str.startswith('M')])
print(df3[df3['Salario'].between(3000, 5000)])
print(df3[df3['departamento'].isin(['Ventas', "Marketing"])])
print(df3.head(2))  # Muestra las primeras 2 filas
print(df3.tail(2))  # Muestra las últimas 2 filas
print(df3.info())  # Información del DataFrame
print(df3.describe())  # Estadísticas descriptivas del DataFrame
print(df3['Salario'].mean())  # Media de la columna 'Salario'
print(df3['Salario'].max())  # Máximo de la columna 'Salario'
print(df3['Salario'].min())  # Mínimo de la columna 'Salario'
print(df3['Salario'].sum())  # Suma de la columna 'Salario'
print(df3['Salario'].count())  # Conteo de la columna 'Salario'
# Accediendo a una columna específica
print(df3['Nombre'].unique())  # Valores únicos en la columna 'Nombre'
print(df3['Nombre'].value_counts())  # Conteo de valores en la columna 'Nombre'
print(df3['Nombre'].nunique)  # Convertir a minúsculas          


         
    
