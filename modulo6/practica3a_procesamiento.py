#Codificacacion variable categórica
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import LabelEncoder 

# 1. Cargar el dataset con variables categóricas y numericas
np.random.seed(42)
df = pd.DataFrame({
    'cliente_id': range(1, 11),
    'genero': np.random.choice(['Masculino', 'Femenino'], size=10),
    'plan': np.random.choice(['Básico', 'Premium', 'VIP'], size=10),
    'estatus': np.random.choice(['Activo', 'Inactivo','pendiente'], size=10),
    'score_crediticio': np.random.randint(300, 850, size=10),
})
print("Dataset original:")
print(df)

#=======================================
# Teoria: Que es Label Encoding 
# Label Encoding convierte las categorías en números enteros.
#Ejemplo: 'Masculino' -> 0, 'Femenino' -> 1 # poner valore categoricos en numeros enteros
#Ventajas:
# - Simple y rápido para categorías ordinales.(donde hay un orden)(bajo, medio, alto)->[0, 1, 2]
#desventajas:
# - No es adecuado para categorías nominales (sin orden) porque introduce un orden artificial.  
# El modelo puede malinterpretar que no hay orden entre los valores.
#2 Aplicación de Label Encoding "Genero" y "estatus"
le_genero = LabelEncoder()
le_estatus = LabelEncoder()
df['genero_encoded'] = le_genero.fit_transform(df['genero']) # Masculino -> 0, Femenino -> 1
df['estatus_encoded'] = le_estatus.fit_transform(df['estatus'])# orden: Activo -> 0, Inactivo -> 1, pendiente -> 2
print("\nDataset con Label Encoding:")
print(df[['cliente_id', 'genero', 'genero_encoded', 'estatus', 'estatus_encoded']])

#=======================================
# Teoria: Que es One Hot Encoding
# One Hot Encoding convierte cada categoría en una columna binaria.
#Ejemplo:  
# plan: 'Básico' -> [1, 0, 0], 'Premium' -> [0, 1, 0], 'VIP' -> [0, 0, 1]
#Ventajas:
#No impone un orden artificial entre categorías, adecuado para categorías nominales.
# - No introduce orden artificial, adecuado para categorías nominales.
# - Permite al modelo aprender relaciones entre categorías.
#Desventajas:
# - Aumenta la dimensionalidad cuando hay muchas categorías.

#3 Aplicación de One Hot Encoding "plan" y "estatus"
columnas = ['plan', 'estatus']
onehot = OneHotEncoder(sparse_output=False, drop=None)  # Sin eliminar columnas

#Crear el transformador de columnas
transformer = ColumnTransformer(
    transformers=[
        ('onehot', onehot, columnas)
    ],
    remainder='passthrough'  # Mantener las columnas restantes
)
#Aplicar la transformación al dataset
datos_transformados = transformer.fit_transform(df)
#obtener los nombres de las nuevas columnas
columnas_onehot = transformer.get_feature_names_out()
# # obtener columnas "passthrough"
# columnas_passthrough = [col for col in df.columns if col not in columnas]
# #Combinar nombres de columnas
# nombre_columnas_finales = list(columnas_onehot) + columnas_passthrough
# #crear un DataFrame con los datos transformados

df_onehot = pd.DataFrame(datos_transformados,columns=columnas_onehot)
print("\nDataset con One Hot Encoding:")
print(df_onehot)

#=======================================
#4. Crear Variables Dummies con pandas
df_dummies = pd.get_dummies(df, columns=['plan', 'estatus'], drop_first=False)
print("\nDataset con Variables Dummies:")
print(df_dummies)


#5. Comparación de métodos
# Label Encoding es simple y rápido, pero puede introducir un orden artificial.
# One Hot Encoding es más adecuado para categorías nominales, pero aumenta la dimensionalidad.
# Variables Dummies en pandas es una forma conveniente de aplicar One Hot Encoding directamente en un DataFrame.
# La elección del método depende del tipo de variable categórica y del modelo que se vaya a utilizar.                   

print("\nComparación de métodos:")
print("Original Dataset:",df.shape)
print("Label Encoding:")
print(df[['cliente_id', 'genero', 'genero_encoded', 'estatus', 'estatus_encoded']].shape)
print("One Hot Encoding:",df_onehot.shape)
print("Variables Dummies:",df_dummies.shape)        


