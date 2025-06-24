import pandas as pd
import numpy as np

#Paso1: Carga los datos
df = pd.read_csv("entradas/ejemplo_datos_outliers.csv")
#Paso 2: inspeccionar rapidamente
print(df.head())
print(df.info())
#Paso3 : Limpiar errores  intencionales(Correccion de texto a NaN)
df["Edad"]= pd.to_numeric(df["Edad"],errors="coerce")
df["Salario"]= pd.to_numeric(df["Salario"],errors="coerce")
df["Horas_Trabajo_Semanal"]= pd.to_numeric(df["Horas_Trabajo_Semanal"],errors="coerce")
#TODO AGREGAR LA DIVISION DE LOS CUARTILES
# df.to_csv("salidas/datos_limpios.csv", index=False, na_rep="NaN")
print(df.info())
#Paso4 : Eliminar duplicados
antes = len(df)
df= df.drop_duplicates()
despues = len(df)
eliminado = antes - despues
df.to_csv("salidas/datos_sin_duplicados.csv", index=False, na_rep="NaN")
print("Filas Antes:",antes)
print("Filas despues:", despues)
print("Filas eliminadas", eliminado)
#Paso5 : Rellenar valores nulos
df["Edad"]=df["Edad"].fillna(df["Edad"].median())
df["Salario"]=df["Salario"].fillna(df["Salario"].mean())
df["Horas_Trabajo_Semanal"]=df["Horas_Trabajo_Semanal"].fillna(df["Horas_Trabajo_Semanal"].mean())
df.to_csv("salidas/datos_rellenados.csv", index=False)
#Paso6  Discretizar "Horas trabajos semanales" en categoria
df["Horas_Categoria"]= pd.cut(df["Horas_Trabajo_Semanal"], bins=[0,30,40,50,np.inf],labels=["Bajo","Normal","Alto","Extremo"])
df.to_csv("salidas/hora_categorizadas.csv",index=False)




