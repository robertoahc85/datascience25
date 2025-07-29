# Zest Analytics, una empresa especializada en el análisis de datos inmobiliarios, te ha
# contratado para desarrollar un modelo de aprendizaje automático que permita predecir
# el precio de viviendas basado en un conjunto de datos recopilado en varias ciudades. El
# conjunto de datos incluye las siguientes características:
# Tamaño en metros cuadrados (size_m2)
# Número de habitaciones
# Antigüedad de la vivienda
# Vecindario
# Precio de venta
# Antes de construir modelos avanzados, el equipo solicita una exploración inicial de los
# datos, su preparación y la construcción de un modelo básico de regresión lineal para
# demostrar su aplicabilidad.

import pandas as pd
import numpy as np
import plotly.express as px 
import plotly.graph_objects as go 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import statsmodels.api as sm 
# from IPpython.display import HTML

#Carga de los datos
df = pd.read_csv("entradas/houses.csv")
#Mostrar la primera 5 fila
print("Primera 5 filas")
print(df.head().to_string)

#Mostra informacion del cojuto de datos
print("Informacion del conjuto de Datos")
print(df.info())

#Mostrar estatidisticas descriptivas
print("Descripcion del cojunto de datos")
print(df.describe())

#2.Limpeza de los datos 
print("Valores faltantes")
print (df.isnull().sum())







