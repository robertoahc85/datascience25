# En un proyecto de construcción, se evalúa la resistencia a la compresión de vigas de
# concreto. Se realizaron 300 ensayos, y la resistencia se distribuye normalmente con una media
# de 30 MPa y una desviación estándar de 2.5 MPa. El estándar de calidad exige que las vigas
# tengan una resistencia mínima de 27 MPa.

#1. Simular 300 valores de resistencia con distribución normal (μ = 30 MPa, σ = 2.5 MPa).
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
#fijas semilla para reproducibilidad
np.random.seed(42)  # Para reproducibilidad
#Paso 1 simular 300 valores
media_resistencia = 30  # MPa
desviacion_estandar = 2.5  # MPa
umbral_minimo = 27  # MPa
n = 30000  # Número de ensayos
resistencias = np.random.normal(media_resistencia, desviacion_estandar, n)

#2. Crear un histograma con una curva KDE para visualizar la distribución.
plt.figure(figsize=(10, 6))
sns.histplot(resistencias, bins=30, kde=True, color='blue', stat='density')
plt.title('Distribución de Resistencia a la Compresión de Vigas de Concreto')
plt.xlabel('Resistencia (MPa)')
plt.ylabel('Densidad')
plt.axvline(x=27, color='red', linestyle='--', label='Límite Mínimo (27 MPa)')
plt.legend()
plt.savefig('output/histograma_resistencia.png')
plt.show()

# 3. Graficar la función de densidad de probabilidad (PDF) teórica.

# 4. Calcular la probabilidad de que una viga tenga resistencia menor a 27 MPa.
#CDF acumulada
prob_menor_27 = stats.norm.cdf(umbral_minimo, loc=media_resistencia, scale=desviacion_estandar)


# 5. Sombrear el área bajo la curva de la PDF para representar esta probabilidad.

# 6. Evaluar si la proporción de vigas con resistencia inferior a 27 MPa representa un riesgo