#Codificacacion variable categórica
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer  
from sklearn.preprocessing import LabelEncoder 
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, Normalizer

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


#-------       

# =============================
# Paso 5: Escalamiento de Variables Numéricas
# =============================

# 🎓 Teoría:
# StandardScaler: centra los datos en 0 y escala con desviación estándar 1.
# Es útil cuando los datos tienen distribución normal. Afectado por outliers.

# MinMaxScaler: escala los datos al rango [0, 1].
# Preserva la forma original de la distribución. Sensible a valores extremos.

# RobustScaler: usa la mediana y el rango intercuartílico (IQR) para escalar.
# Útil cuando hay outliers que distorsionarían las otras técnicas.

# Definir diccionario con escaladores
scalers = {
    "StandardScaler": StandardScaler(),     # Centrado en 0, varianza 1
    "MinMaxScaler": MinMaxScaler(),         # Rango [0, 1]
    "RobustScaler": RobustScaler()          # Mediana = 0, escala por IQR
}

# Aplicar cada escalador y crear nuevas columnas
for name, scaler in scalers.items():
    df[f'score_{name}'] = scaler.fit_transform(df[['score_crediticio']])

# =============================
# Paso 6: Transformaciones Matemáticas
# =============================

# Logaritmo natural de (1 + x) para evitar problemas con log(0)
df['score_log'] = np.log1p(df['score_crediticio'])

# Raíz cuadrada
df['score_sqrt'] = np.sqrt(df['score_crediticio'])

# Box-Cox (Yeo-Johnson) para mejorar normalidad
pt = PowerTransformer(method='yeo-johnson')
df['score_boxcox'] = pt.fit_transform(df[['score_crediticio']])

# =============================
# Paso 7: Normalización
# =============================

# Normaliza los datos para que tengan norma L2 = 1 (cada fila se escala)
normalizer = Normalizer()
df['score_normalized'] = normalizer.fit_transform(df[['score_crediticio']])

# ===============================================
# Procesamiento y Visualización de Datos Escalados
# ===============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, StandardScaler,
    MinMaxScaler, RobustScaler, PowerTransformer, Normalizer
)
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ===============================
# Paso 1: Dataset simulado
# ===============================

np.random.seed(42)
df = pd.DataFrame({
    'cliente_id': range(1, 11),
    'genero': np.random.choice(['Masculino', 'Femenino'], size=10),
    'plan': np.random.choice(['Básico', 'Premium', 'VIP'], size=10),
    'estatus': np.random.choice(['Activo', 'Inactivo','Pendiente'], size=10),
    'score_crediticio': np.random.randint(300, 850, size=10)
})

# ===============================
# Paso 2: Label Encoding
# ===============================

le_genero = LabelEncoder()
le_estatus = LabelEncoder()

df['genero_encoded'] = le_genero.fit_transform(df['genero'])
df['estatus_encoded'] = le_estatus.fit_transform(df['estatus'])

# ===============================
# Paso 3: One-Hot Encoding
# ===============================

columnas = ['plan', 'estatus']
onehot = OneHotEncoder(sparse_output=False, drop=None)

transformer = ColumnTransformer(
    transformers=[('onehot', onehot, columnas)],
    remainder='passthrough'
)

datos_transformados = transformer.fit_transform(df)
columnas_transformadas = transformer.get_feature_names_out()
df_onehot = pd.DataFrame(datos_transformados, columns=columnas_transformadas)

# ===============================
# Paso 4: Dummies con pandas
# ===============================

df_dummies = pd.get_dummies(df, columns=['plan', 'estatus'], drop_first=False)

# ===============================
# Paso 5: Escalamiento de score_crediticio #Normalizacion

#standardScaler : Centra los datos en 0 y escala con desviación estándar 1.
#Es útil cuando los datos tienen distribución normal. Afectado por outliers.

# MinMaxScaler: Escala los datos al rango [0, 1].
# Preserva la forma original de la distribución. Sensible a valores extremos.

# RobustScaler: Usa la mediana y el rango intercuartílico (IQR) para escalar.
# Útil cuando hay outliers que distorsionarían las otras técnicas.
# ===============================
#Definir diccionario con escaladores
scalers = {
    "StandardScaler": StandardScaler(),     # Centrado en 0, varianza 1
    "MinMaxScaler": MinMaxScaler(),         # Rango [0, 1]
    "RobustScaler": RobustScaler()          # Mediana = 0, escala por IQR
}
# Aplicar cada escalador y crear nuevas columnas
for name, scaler in scalers.items():
    scaled_values= df[f'score_{name}'] = scaler.fit_transform(df[['score_crediticio']])
    df[f'score_{name}'] = scaled_values
    print(f"score_{name} aplicado:")
    
 #imprimir los valores escalados
print("\n DataFrame con valores escalados:")  
print(df[['cliente_id', 'score_crediticio'] + [f'score_{name}' for name in scalers.keys()]]) 
# MinMaxScaler  siempre dara valores entre 0 y 1
# StandardScaler dara valores centrados en 0
# RobustScaler generara valores centrados en la mediana y menos afectados por outliers

#El objetivo mejorar el rendimiento de los modelos de machine learning.(Regresion lines, knn, SVM)

#Paso 6 Transformaciones Matemáticas
# Logaritmo natural de (1 + x) para evitar problemas con log(0)
df['score_log'] = np.log1p(df['score_crediticio']) # el crecimiento se aplana, los altos se reduce propocinal  mas que lo bajos
# Raíz cuadrada
df['score_sqrt'] = np.sqrt(df['score_crediticio']) # util  cuando los datos no son tan sesgados y los valores moderamenta disperso
# Box-Cox (Yeo-Johnson) para mejorar normalidad
pt = PowerTransformer(method='yeo-johnson')
df['score_boxcox'] = pt.fit_transform(df[['score_crediticio']]) # busca cambiar la distribucion de los datos para que se asemeje a una normal, es decir, mejorar la normalidad de los datos
#imprimir DataFrame 
print("\n DataFrame con transformaciones matemáticas:")  
print(df[['cliente_id', 'score_crediticio', 'score_log', 'score_sqrt', 'score_boxcox']])


# Paso 7: Normalización
# Normaliza los datos para que tengan norma L2 = 1 (cada fila se escala)
normalizer = Normalizer(norm='l2') # Normaliza cada fila para que su norma L2 sea igual a 1
cols_to_normalize = ['score_crediticio']
normalized_array = normalizer.fit_transform(df[['score_crediticio']]) #Normalizacion ecluadinna, Tranforma  un vector de datos para que su norma L2 sea igual a 1, es decir, la suma de los cuadrados de sus componentes sea igual a 1
df_normalized = pd.DataFrame(normalized_array, columns=[f'{col}_l2' for  col in cols_to_normalize])
df_final = pd.concat([df, df_normalized], axis=1)
print("\n DataFrame con normalización L2:")  
print(df_final)

#L2 cuano solo normalizamos una columna, todos los valores son 1.0

# Normalizar para  sobre muiltiples columnas numericas
cols_to_normalize = ['score_crediticio', 'score_log', 'score_sqrt', 'score_boxcox']
normalize = Normalizer(norm='l2') # Normaliza cada fila para que su norma L2 sea igual a 1
normalized_array = normalize.fit_transform(df[cols_to_normalize])
df_normalized = pd.DataFrame(normalized_array, columns=[f'{col}_l2' for col in cols_to_normalize])
df_final = pd.concat([df, df_normalized], axis=1)
print("\n DataFrame con normalización L2 en múltiples columnas:")  
print(df_final[['cliente_id'] + [f'{col}_l2' for col in cols_to_normalize]])

#Paso 8 Visualización de Datos Escalados
# Visualizar la distribución de las variables escaladas
plt.figure(figsize=(12, 8))
for  name in scalers:
    sns.kdeplot(df[f'score_{name}'], label=f'Score {name}', fill=True)
plt.title('Distribución de Scores Escalados')
plt.xlabel('Score Crediticio Escalado')
plt.ylabel('Densidad')
plt.legend()
plt.tight_layout()
os.makedirs('dashboard', exist_ok=True)
plt.savefig('dashboard/distribucion_scores_escalados.png')
plt.close()

#paso 9: Resumen de Técnicas de Procesamiento
# Crear un resumen de las técnicas aplicadas y sus estadísticas
# Guardar el resumen en un archivo CSV
resumen = pd.DataFrame({
    "Tecnica": ["Label Encoding", "One Hot Encoding", "Variables Dummies", "StandardScaler", "MinMaxScaler", "RobustScaler", "PowerTransformer", "Normalizer"],
    "Media": [df['score_crediticio'].mean(), 
        df_onehot.filter(like='plan_').mean().mean(), 
        df_dummies.filter(like='plan_').mean().mean(),
        df['score_StandardScaler'].mean(),
        df['score_MinMaxScaler'].mean(),
        df['score_RobustScaler'].mean(),
        df['score_boxcox'].mean(),
        df_final.filter(like='_l2').mean().mean() 
    ],"Desviación Estándar": [
        df['score_crediticio'].std(), 
        df_onehot.filter(like='plan_').std().mean(), 
        df_dummies.filter(like='plan_').std().mean(),
        df['score_StandardScaler'].std(),
        df['score_MinMaxScaler'].std(),
        df['score_RobustScaler'].std(),
        df['score_boxcox'].std(),
        df_final.filter(like='_l2').std().mean()
    ]
})

resumen.to_csv('dashboard/resumen_tecnicas.csv', index=False)

#Paso 10" interpretación  de resultado en html
interpretacion_html = "<div style='font-family: Arial, sans-serif; line-height: 1.6;'> <h2>Interpretación de Resultados Automatica</h2>" 
for _, row in resumen.iterrows():
    tecnica = row['Tecnica']
    media = row['Media']
    desviacion = row['Desviación Estándar']
    if tecnica == "Original":
        msg = f"<p><strong>{tecnica}</strong>: Media = {media:.2f}, Desviación Estándar = {desviacion:.2f} (Valores originales)</p>"
    elif tecnica == "Label Encoding":
        msg = f"<p><strong>{tecnica}</strong>: Media = {media:.2f}, Desviación Estándar = {desviacion:.2f} (Valores enteros asignados a categorías)</p>"
    elif tecnica in ["One Hot Encoding", "Variables Dummies"]:
        msg = f"<p><strong>{tecnica}</strong>: Media = {media:.2f}, Desviación Estándar = {desviacion:.2f} (Valores binarios asignados a categorías)</p>"
    elif tecnica in ["StandardScaler", "MinMaxScaler", "RobustScaler", "PowerTransformer", "Normalizer"]: 
        msg = f"<p><strong>{tecnica}</strong>: Media = {media:.2f}, Desviación Estándar = {desviacion:.2f} (Valores escalados)</p>"                 
        
interpretacion_html +=f"<li> msg</li> "      
interpretacion_html += "</ul></div>"

with open('dashboard/interpretacion_resultados.html', 'w') as f:
    f.write(interpretacion_html)   
#Paso 11: Generación de Dashboard html completo
# Convertimos resumen a HTML
resumen_html = resumen.to_html(index=False, classes='table table-bordered table-striped', border=0)

# Combinamos todo en el dashboard
dashboard_html = f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Dashboard de Procesamiento de Datos</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-5">
        <h1 class="mb-4">Dashboard de Procesamiento de Datos</h1>
        <p>Este dashboard resume el procesamiento de variables categóricas y numéricas utilizando técnicas modernas.</p>

        <h2>📊 Distribución de Scores Escalados</h2>
        <img src="distribucion_scores_escalados.png" class="img-fluid rounded shadow">

        <h2 class="mt-5">📄 Resumen de Técnicas Aplicadas</h2>
        {resumen_html}

        <h2 class="mt-5">📌 Interpretación Automática</h2>
        {interpretacion_html}
    </div>
</body>
</html>
"""

# Guardamos el archivo final
with open("dashboard/dashboard.html", "w", encoding='utf-8') as f:
    f.write(dashboard_html)
