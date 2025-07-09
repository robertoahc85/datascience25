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
#Regresión lineal con interacción
import statsmodels.api as sm
import statsmodels.formula.api as smf
sns.set(style="whitegrid")

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
A = df["tip"] >= 5 # Evento A: Propinas mayores
B = df["day"] == "Sun"  # Evento B: Consumo realizado
# Calcular probabilidades simples
P_A = A.mean()  # Probabilidad de A P(A)= numero de propinas mayores a 5 / total de propinas
P_B = B.mean()  # Probabilidad de B P(B)= numero de consumos realizados en domingo / total de consumos
print(f"Probabilidad de A (Propinas mayores a 5): {P_A:.2f}")
print(f"Probabilidad de B (Consumo realizado en domingo): {P_B:.2f}")   
# Calcular probabilidades conjuntas
P_AyB = (A & B).mean()  # Probabilidad de A y B #P(A ∩ B)= numero de propinas mayores a 5 y consumos realizados en domingo / total de consumos 
print(f"Probabilidad de A y B (Propinas mayores a 5 y consumo en domingo): {P_AyB:.2f}")
P_A_B = (A & B).sum()/B.sum() # Probabilidad de A dado B #P(A | B) = numero de propina alta en Domingp/ Total de domingo
print(f"Probabilidad de A dado B (Propinas mayores a 5 dado consumo en domingo): {P_A_B:.2f}")
# Visualizar la diferencia entre el experimento determinístico y aleatorio
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='Experiment', y='tip', data=df)
# plt.title('Comparación de Propinas: Determinístico vs Aleatorio')
# plt.xlabel('Experimento')
# plt.ylabel('Propina')
# plt.grid(True)
# plt.savefig("output/boxplot_propinas.png")
# plt.show()

#3.1 Tabla (arbol) de probabilidades
tabla = (df.assign(tip_bin=pd.cut(df['tip'], bins=[0, 5, np.inf], labels=['<=5', '>5']))
         .groupby(['Experiment', 'tip_bin'])
         .size()
         .div(len(df)) # = n * de casos en esa combinación / total de casos
         .unstack(fill_value=0) # un tabla data frame con los valores de la tabla de probabilidades
)
print("\nTabla de Probabilidades: \n", tabla)

#Efecto del muestreo en la estimación de probabilidades
# Muestreo aleatorio
muestra_aleatorio = df.sample(60, random_state=0)
muestra_sesgado = df.query("time == 'Dinner'").sample(60, random_state=0)
print("\nMuestra Aleatoria:\n", muestra_aleatorio.head())
print("\nMuestra Sesgada:\n", muestra_sesgado.head())
print("\nMedia de la propina - muestra Aleatorioa:", muestra_aleatorio['tip'].mean())
print("\nMedia de la propina - muestra Sesgada:", muestra_sesgado['tip'].mean())

#Regresión lineal con interacción
# Modelo de regresión lineal con interacción
#Modelo : tip= β0 + β1·total_bill + β2·Experiment + β3·total_bill*Experiment
model = smf.ols('tip ~ total_bill * Experiment', data=df).fit() #regresión lineal por mínimos cuadrados
# print("\nResumen del modelo de regresión lineal:\n", model.summary().tables[0]) #table[0] Metadato del modelo r2,n,etc,
print("\nResumen del modelo de regresión lineal:\n", model.summary().tables[1]) #table[1] coeficientes de regressión 
# print("\nResumen del modelo de regresión lineal:\n", model.summary().tables[2]) #table[2] Informacion extra
# Visualizar los resultados de la regresión

# Visualizar los resultados de la regresión
sns.lmplot(x='total_bill', y='tip', hue='Experiment', data=df,
           height=6, aspect=1.3,
           scatter_kws={'alpha': 0.45})
plt.title('Propina idea (15%) vs Propina real (aleatoria)')
plt.show()



