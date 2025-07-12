# Escenario: Una tienda vende bebidas con un volumen promedio de 500 ml
# y una desviación estándar de 30 ml (distribución no especificada). 
# Simula la distribución muestral de la media para muestras de n = 36.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar semilla para reproducibilidad
np.random.seed(123)

#Parametros de la distribucion
mu = 500  #    Media de la población
sigma = 30  # Desviación estándar de la población
n = 36  # Tamaño de la muestra
num_muestras = 1000  # Número de muestras a simular

#Generar medias muestral
medias_muestrales = [np.mean(np.random.normal(mu, sigma, n)) for _ in range(num_muestras)]
#calcular media y desviación estándar de la distribución muestral
error_standar = np.std(medias_muestrales,ddof=0)
error_teorico = sigma / np.sqrt(n)
print("Media de la distribución muestral:", np.mean(medias_muestrales))
print("Error standar:", error_standar)      
print("Error estándar teórico:", error_teorico)

#visualizar la distribución muestral
plt.figure(figsize=(10, 6))
sns.histplot(medias_muestrales, bins=30, kde=True, color='blue')
plt.axvline(mu, color='red', linestyle='dashed', linewidth=2, label='Media poblacional (500 ml)')
plt.title('Distribución Muestral de la Media')
plt.xlabel('Media Muestral (ml)')
plt.ylabel('Frecuencia')
plt.legend()
plt.show()      
plt.savefig('graficas/distribucion_muestral_media.png', dpi=300, bbox_inches='tight')  # Guardar la figura




