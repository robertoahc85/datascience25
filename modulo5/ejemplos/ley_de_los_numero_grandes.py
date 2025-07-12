import numpy as np
import matplotlib.pyplot as plt

# Configurar la semilla para reproducibilidad
np.random.seed(42)

# Parámetros de la población (distribución exponencial)
lambda_param = 1 / 5  # Parámetro de la distribución exponencial (1 / media)
mu = 5  # Media poblacional teórica (1 / lambda)

# Tamaños de muestra a probar
tamanos_muestra = [10, 100, 1000, 10000]

# Almacenar las medias muestrales para cada tamaño de muestra
medias_por_tamano = []

# Generar muestras y calcular medias para cada tamaño
for n in tamanos_muestra:
    # Generar 1000 muestras de tamaño n y calcular sus medias
    medias_muestrales = [np.mean(np.random.exponential(scale=mu, size=n)) for _ in range(1000)]
    medias_por_tamano.append(medias_muestrales)

# Visualizar los resultados con un diagrama de caja (boxplot)
plt.figure(figsize=(10, 6))
plt.boxplot(medias_por_tamano, labels=tamanos_muestra)
plt.axhline(y=mu, color='red', linestyle='--', label='Media Poblacional (5 minutos)')
plt.title('Ley de los Grandes Números: Convergencia de la Media Muestral')
plt.xlabel('Tamaño de la Muestra (n)')
plt.ylabel('Media Muestral (minutos)')
plt.legend()
plt.show()

# Imprimir las medias de las medias muestrales para cada tamaño
for i, n in enumerate(tamanos_muestra):
    print(f"Tamaño de muestra n={n}: Media de las medias muestrales = {np.mean(medias_por_tamano[i]):.2f} minutos")