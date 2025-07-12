import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar la semilla para reproducibilidad
np.random.seed(42)

# Parámetros de la población
mu = 150  # Media poblacional (gramos)
sigma = 20  # Desviación estándar poblacional (gramos)
n = 36  # Tamaño de la muestra
num_muestras = 1000  # Número de muestras

# Generar muestras aleatorias y calcular medias muestrales
medias_muestrales = []
for _ in range(num_muestras):
    muestra = np.random.normal(mu, sigma, n)  # Generar muestra de tamaño n
    media_muestral = np.mean(muestra)  # Calcular la media de la muestra
    medias_muestrales.append(media_muestral)

# Calcular propiedades de la distribución muestral
media_dist_muestral = np.mean(medias_muestrales)  # Media de las medias muestrales
error_estandar = np.std(medias_muestrales, ddof=1)  # Error estándar estimado

# Imprimir resultados
print(f"Media de la distribución muestral: {media_dist_muestral:.2f} gramos")
print(f"Error estándar de la distribución muestral: {error_estandar:.2f} gramos")
print(f"Error estándar teórico: {sigma / np.sqrt(n):.2f} gramos")

# Visualizar la distribución muestral
plt.figure(figsize=(10, 6))
sns.histplot(medias_muestrales, kde=True, color='blue', bins=30)
plt.title('Distribución Muestral de la Media (n=36, 1000 muestras)')
plt.xlabel('Media Muestral (gramos)')
plt.ylabel('Frecuencia')
plt.axvline(mu, color='red', linestyle='--', label='Media Poblacional (150 g)')
plt.legend()
plt.show()