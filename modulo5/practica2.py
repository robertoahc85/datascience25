# Objetivo: Mostrar, con un caso real de restaurantes, cómo la Teoría de Probabilidades permite:

# Definir un espacio muestral y eventos medibles.

# Calcular probabilidades simples, conjuntas y condicionales.

# Visualizar y cuantificar la diferencia entre un experimento determinístico 
# (propina “ideal” = 15 % de la cuenta) y un experimento aleatorio (propinas reales, sujetas a variabilidad humana).

# Analizar el impacto del muestreo (aleatorio vs sesgado)
# sobre las estimaciones.

# Integrar todo en un modelo de regresión lineal con 
# interacción para contrastar relaciones “perfectas” vs “ruidosas”.

import pandas as pd,numpy as np ,matplotlib.pyplot as plt, seaborn as sns

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
tips = pd.read_csv(url)
#Guardar el DataFrame en un archivo CSV
tips.to_csv("input/tips_original.csv", index=False)

#Construir  dos grupos de observaciones
# Grupo 1 Propina ideal (determinístico) = 15% de la cuenta
tips_det = tips.copy()
tips_det['tip'] = (tips_det['total_bill'] * 0.15).round(2)  # Redondear a dos decimales
tips_det['Experiment'] = "Deterministico"

#Grupo 2 propina real (observaciones reales)
tips_ran = tips.copy()
tips_ran['Experiment'] = "Aleatorio" #Fenomeno con incertidumbre

#Unir ambos grupos
df = pd.concat([tips_det, tips_ran], ignore_index=True)
#Guardar el DataFrame combinado en un archivo CSV
df.to_csv("input/tips_combined.csv", index=False)

#Probalidad Basica
#Definir el espacio muestral y eventos medibles
#Espacio muestral: Todas las observaciones de propinas
# A: Eventos medibles: Propinas mayores a 5
# B: Eventos medibles: consumo realizado en domingo (sun)
S = df["Experiment"].unique  # # Espacio muestral
A = df[df["tip"] > 5]  # Evento A: Propinas mayores
B = df[df["day"] == "Sun"]  # Evento B: Consumo realizado
# Calcular probabilidades simples
P_A = A.mean()  # Probabilidad de A
P_B = B.mean()  # Probabilidad de B
print(P_A)
# Calcular probabilidades conjuntas
P_AyB = (A & B).mean()  # Probabilidad de A y B #P(A ∩ B)
P_A_B = (P & B).sum()/B.sum() # Probabilidad de A dado B #P(A | B)








