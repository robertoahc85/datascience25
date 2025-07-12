import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar la semilla para reproducibilidad
np.random.seed(42)

# Parámetros de la población (distribución exponencial)
lambda_param = 1 / 5  # Parámetro de la distribución exponencial (1 / media)
mu = 5  # Media poblacional teórica
sigma = 5  # Desviación estándar poblacional (para exponencial, sigma = mu)

# Tamaños de muestra a probar
tamanos_muestra = [5, 10, 30, 50]
num_muestras = 1000  # Número de muestras por cada tamaño

# Generar y almacenar medias muestrales para cada tamaño de muestra
medias_por_tamano = {}
for n in tamanos_muestra:
    medias_muestrales = [np.mean(np.random.exponential(scale=mu, size=n)) for _ in range(num_muestras)]
    medias_por_tamano[n] = medias_muestrales

# Visualizar las distribuciones muestrales
plt.figure(figsize=(12, 8))
for i, n in enumerate(tamanos_muestra, 1):
    plt.subplot(2, 2, i)
    sns.histplot(medias_por_tamano[n], kde=True, color='blue', bins=30)
    plt.title(f'Distribución Muestral (n={n})')
    plt.xlabel('Media Muestral (minutos)')
    plt.ylabel('Frecuencia')
    plt.axvline(mu, color='red', linestyle='--', label='Media Poblacional (5 min)')
    plt.legend()

plt.tight_layout()
plt.show()

# Imprimir propiedades de la distribución muestral
for n in tamanos_muestra:
    media_dist_muestral = np.mean(medias_por_tamano[n])
    error_estandar = np.std(medias_por_tamano[n], ddof=1)
    error_teorico = sigma / np.sqrt(n)
    print(f"Tamaño de muestra n={n}:")
    print(f"  Media de la distribución muestral: {media_dist_muestral:.2f} minutos")
    print(f"  Error estándar estimado: {error_estandar:.2f} minutos")
    print(f"  Error estándar teórico: {error_teorico:.2f} minutos")