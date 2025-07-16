# ---------------------------------------------
# Ejercicio Adicional 2: Distribuciones Muestrales - Medias Muestrales
# Contexto: Un climatólogo analiza la precipitación promedio mensual en una región.
# Entrada: 500 simulaciones de muestras de 40 días con precipitación media de 100 mm
# y desviación estándar de 15 mm.
# Salida esperada: Distribución de las medias muestrales.
# Pasos:
# 1. Identificar datos de entrada.
# 2. Simular medias muestrales con distribución normal.
# 3. Calcular media y desviación estándar de las medias muestrales.
# 4. Verificar el teorema del límite central.
# 5. Visualizar con histograma.

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

n = 40 #Tamano de la muestra
simulacion = 500 #numero de simulaciones
pop_mean =100
pop_std = 15 #Desviacion standar poblacional mm

#Simular medias muestrales
sample_mean = [np.mean(np.random.normal(pop_mean,pop_std,n)) for _ in range(simulacion)]
mean_sample_mean = np.mean(sample_mean)
std_sample_means = np.std(sample_mean)

print("---Distribuccion muestral--")
print(f"Media de las medias muestrales: {mean_sample_mean:.2f}")
print(f"Desviación estándar de las medias muestrales: {std_sample_means:.2f}")

# Visualizar el histograma de las medias muestrales
plt.figure(figsize=(10,6))
plt.hist(sample_mean, bins=20, alpha=0.7, color='lightcoral', edgecolor='black', density=True)
plt.axvline(pop_mean, color='red', linestyle='--',label='Media poblacional')
plt.title('Distribuccion medias muestra de precipitacion')
plt.xlabel('Media muestral de precipitación (mm)')
plt.ylabel('Densidad')
sns.kdeplot(sample_mean, color='blue', label='KDE')
plt.legend()
plt.show()
plt.savefig('graficas/distribucion_muestral2.png')





