#Practicas de series de pandas
import pandas as pd
# #entrada
# sales = [100, 200, 1300, 400, 1500,1801, 2000, 3000, 4000, 5000]
# months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre']
# series = pd.Series(sales, index=months)
# print(series)

# #Salida
# #Enero: 100
# #Febrero: 200
# #Marzo: 1300..

# #Entrada stock Frutas
# stock = {"Manzanas": 50, "Peras": 30, "Naranjas": 20, "Platanos": 15,"Fresas": 25, "Kiwi": 10 ,"Uvas": 5 , "Sandia": 8}
# stock_series = pd.Series(stock)
# print(stock_series)

# #salida  inventario frutas con su stock

# #Entrada dia y temperatura
# dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
# temperaturas = [20, 22, 19, 21, 23, 25, 24]
# temperaturas_series = pd.Series(temperaturas, index=dias)
# print(temperaturas_series)

# #salida lista de temperaturas por dia

# #Entrada rating y customer
# ratings = ['Bueno', 3.8, 'Excelente', 4.0]
# customers = ['Cliente1', 'Cliente2', 'Cliente3', 'Cliente4']
# series_ratings = pd.Series(ratings, index=customers)
# print(series_ratings) 

#Acceder a las series

sales = pd.Series([100, 200, 1300, 400, 1500, 1801, 2000, 3000, 4000, 5000], index=['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre'])
print(sales['Febrero'])  # Acceder al valor de Febrero
print(sales[['Enero', 'Marzo']])  # Acceder a varios valores
# print(sales[0])  # Acceder al primer valor (Enero)
print(sales[1:4])  # Acceder a un rango de valores (Febrero a Abril)
print(sales.values)
print("Venta mayores a 1400:")
print(sales[sales > 1400 ])  # Acceder al índice de la serie
#Acceder a los valores de la serie   
print("Multiplicar los valores de la serie por 2:")
print( sales * 2)  # Multiplicar todos los valores por 2
print("subir el iva a 19% a las ventas:")
print(sales * 1.19)  # Subir el IVA a 19% a las ventas
# Calcular la media de las ventas
print("Media de las ventas:")
print(sales.mean())
# Calcular la desviación estándar de las ventas
print("Desviación estándar de las ventas:")
print(sales.std())
#calcular la suma de las ventas
print("Suma de las ventas:")
print(sales.sum())  # Sumar todos los valores de la serie
#calular el valor máximo de las ventas
print("Valor máximo de las ventas:")
print(sales.max())  # Valor máximo de la serie
#calcular el valor mínimo de las ventas
print("Valor mínimo de las ventas:")
print(sales.min())  # Valor mínimo de la serie



#Acceder inidice