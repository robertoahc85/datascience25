#Analisis Variables Aleatorias y Distribuciones con python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
from scipy.stats  import poisson
# Configuración de estilo para los gráficos

#Generacion de datos simulados
#fijmos la semilla para reproducibilidad
np.random.seed(42)
#Numero de registros simulados
n = 300
#Variables discretas ,numero de llamadas ,simuladas con distribucion Poisson
llamadas_por_dia = np.random.poisson(lam=5, size=n)

#Variables continuas, tiempos de espera simulados con distribucion normal
duracion_llamada= np.random.normal(loc=8, scale=2, size=n)

#variables categoricas, tipo de llamada simuladas con distribucion uniforme
nivel_satisfacion= np.random.randint(1, 6, size=n) # # 1 a 5 el ultimo valor es 5 por que el 6 excluido

df = pd.DataFrame({
    'llamadas': llamadas_por_dia,
    'duracion_llamada': np.round(duracion_llamada, 2),  # Redondear a 2 decimales
    'nivel_satisfacion': nivel_satisfacion
})
df.to_csv('input/datos_llamadas.csv', index=False)

# Distribuccion discreta: llamadas por dia
frencuencias_llamadas = df['llamadas'].value_counts().sort_index()
print("Frecuencias de llamadas por día:")
print(frencuencias_llamadas)
frencuencias_llamadas.plot(kind='bar', color='skyblue')
plt.title('Frecuencia de llamadas por día')
plt.xlabel('Número de llamadas')
plt.ylabel('Frecuencia dias')               
plt.grid(axis='y')
plt.tight_layout()    
plt.savefig('output/frecuencia_llamadas_por_dia.png')
# plt.show()  

# Distribucion continua: duracion de llamadas
#Histrograma  y KDE VARIABLES CONTINUAS
sns.histplot(df['duracion_llamada'], bins=30, kde=True, color='lightgreen')
plt.title('Histograma de la duración de llamadas')
plt.xlabel('Duración de llamada (minutos)')
plt.ylabel('Frecuencia')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('output/histograma_duracion_llamadas.png')
# plt.show()

# Funcion de densidad de la normal (Formal funcional)
x_vals = np.linspace(0,16,200)
y_vals = stats.norm.pdf(x_vals, loc=8, scale=2)

#Graficar la funcion de densidad
plt.plot(x_vals, y_vals, color='blue', label='Función de densidad')
plt.title('Función de la distribución normal') 
plt.xlabel('Duración de llamada (minutos)')
plt.ylabel('Densidad')
plt.grid(True)  
plt.tight_layout()
plt.legend()
plt.savefig('output/funcion_densidad_normal.png')
# plt.show()                   

# cual es la probabilidad de que la duracion de una llamada sea 7 llamada al dia
prob_7 = poisson.pmf(7, mu=5)
print(f'Probabilidad de recibir 7 llamadas en un día: {prob_7:.4f}')

# cual es la probabilidad de que la duracion de una llamada sea menor a 7 minutos
prob_menor_7 = stats.norm.cdf(7, loc=8, scale=2)
print(f'Probabilidad de que la duración de una llamada sea menor a 7 minutos: {prob_menor_7:.4f}')

