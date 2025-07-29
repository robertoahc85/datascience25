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

#inputar Valores faltante numerico con la mediana 
for column  in ['size_m2','bedrooms','age','price']:
    if df[column].isnull().sum() > 0:
        df[column].fillna(df[column].median(),inplace=True)

df['size_m2'] =df['size_m2'].astype(float)
df['bedrooms'] =df['bedrooms'].astype(int)
df['age'] =df['age'].astype(int)
df['price'] =df['price'].astype(float)
df['neighborhood'] =df['neighborhood'].astype(str)

#Verifcar que no queden valores faltante
print("Valores faltantes despues del imputacion")
print (df.isnull().sum())

# 3. Analisis Exploratorios de Datos
#Grafico interactivo: Relaxcion entre tamano y precio
fig1 = px.scatter(
    df,
    x='size_m2',
    y='price',
    color='neighborhood',
    title="Tamaño de la vivienda vs Precio",
    labels={'size_m2': 'Tamaño (m2)', 'price': 'Precio'},
    hover_data= ['bedrooms','age'])
fig1.update_layout(showlegend = True)
fig1.write_html('graph/size_vs_price.html')
# fig1.show()

#Grafico interactivo : Relacion entre numero de habitacion y precio
fig2 = px.box(df,x='bedrooms',y ='price', color='neighborhood',
              title = "Numero de habitacion vs Precio",
              labels={'bedrooms': 'Habitaciones', 'price': 'Precio'})
fig2.update_layout(showlegend = True)
fig2.write_html('graph/bedrooms_vs_price.html')


#Detectar valores atipico (outlier)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['price'] < (Q1 - 1.5 * IQR)) | (df['price'] > (Q3 + 1.5 * IQR))]
print(f"Numero de valores atipicos en precio: {len(outliers)}")

# 4. Codificacion de variables categoricas
# #Aplicar en One-Hot en la columna neighboord
encoder = OneHotEncoder(sparse=False, drop='first')
#Aplica la codificacion al Dataframe original
encoded_neighborhood  = encoder.fit_transform(df[['neighborhood']])
#Convierte el resultado un nuevo dataframe con nombre de columnas claros
encoded_df = pd.DataFrame(encoded_neighborhood, columns= encoder.get_feature_name({'neighborhood'}))

#Combina columnas codificadas. 
df = pd.concat([df.drop('neighborhood',axis=1),encoded_df],axis=1)





        






