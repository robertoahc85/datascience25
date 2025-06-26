import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Paso 1. Cargo de datos
df=pd.read_csv("entrada/empleados_150.csv")

#Paso 2 .Inspeccion  Inicial
print("---------Primera filas del dataframe")
print(df.head(10))
print(df.describe())
print("+++++++++Informacion del dataframe")
print(df.info())

#Paso3 . Indentificacion de problemas 
print("Valor faltantes por columna")
print(df.isnull().sum())
print("Filas duplicadas",df.duplicated().sum())
print("Tipo de datos por columna")
print(df.dtypes)

#Paso4  Limpieza de lsod