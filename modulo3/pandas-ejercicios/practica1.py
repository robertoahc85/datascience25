import pandas as pd
#entrada
sales = [100, 200, 1300, 400, 1500,1801, 2000, 3000, 4000, 5000]
months = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 'Julio', 'Agosto', 'Septiembre', 'Octubre']
series = pd.Series(sales, index=months)
print(series)

#Salida
#Enero: 100
#Febrero: 200
#Marzo: 1300..

#Entrada stock Frutas
stock = {"Manzanas": 50, "Peras": 30, "Naranjas": 20, "Platanos": 15,"Fresas": 25, "Kiwi": 10 ,"Uvas": 5 , "Sandia": 8}
stock_series = pd.Series(stock)
print(stock_series)

#salida  inventario frutas con su stock

#Entrada dia y temperatura
dias = ['Lunes', 'Martes', 'Miercoles', 'Jueves', 'Viernes', 'Sabado', 'Domingo']
temperaturas = [20, 22, 19, 21, 23, 25, 24]
temperaturas_series = pd.Series(temperaturas, index=dias)
print(temperaturas_series)

#salida lista de temperaturas por dia