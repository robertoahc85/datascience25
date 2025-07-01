import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar el archivo
df = pd.read_csv("entrada/student_performance.csv")
#Imprimir la primera 5 filas
print("Primeras 5 filas ")
print(df.head(5))

#2.Analisis Corelacion (Pearson)
numerics_vars = ['hours_studied','exam_score','coffee_cups']
cor_matrix = df[numerics_vars].corr(method='pearson')

print("\nMatriz de corelacion de Pearson")
print(cor_matrix)

#Alternativa  , podamos calcular entre dos variable usando Numpy
pearson_r = np.corrcoef(df['hours_studied'],df['exam_score'])[0,1]
print(f"\nCoeficiente de Pearson entre horas de estudios y puntaje:{pearson_r:.3f}")

#Tabla de contigencia entre genero y grupo de estudio
crosstab_result = pd.crosstab(df['gender'], df['study_group'],)
print("\nTabla de contigencia entre genero y grupo de estudio")
print(crosstab_result)

#4. Visualizacion de Corelaciones





