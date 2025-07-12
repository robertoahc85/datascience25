import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Configurar la semilla para reproducibilidad
np.random.seed(42)

# Parámetros de la población y la muestra
p = 0.6  # Proporción poblacional
n = 100  # Tamaño de la muestra
num_muestras = 1000  # Número de muestras para simulación

# 1. Simulación de la distribución muestral
proporciones_muestrales = []
for _ in range(num_muestras):
    muestra = np.random.binomial(n, p)  # Número de éxitos en n ensayos
    prop_muestral = muestra / n  # Proporción muestral
    proporciones_muestrales.append(prop_muestral)

# Calcular el error estándar
error_estandar = np.sqrt(p * (1 - p) / n)

# 2. Probabilidad de que la proporción muestral sea mayor a 0.65
p1 = 0.65
z1 = (p1 - p) / error_estandar
prob_mayor_065 = 1 - norm.cdf(z1)

# 3. Probabilidad de que la proporción muestral esté entre 0.55 y 0.65
p2_lower = 0.55
p2_upper = 0.65
z2_lower = (p2_lower - p) / error_estandar
z2_upper = (p2_upper - p) / error_estandar
prob_entre_055_065 = norm.cdf(z2_upper) - norm.cdf(z2_lower)

# Imprimir resultados
print(f"Error estándar: {error_estandar:.4f}")
print(f"Media simulada de la distribución muestral: {np.mean(proporciones_muestrales):.4f}")
print(f"Probabilidad de que la proporción muestral sea mayor a 0.65: {prob_mayor_065:.4f}")
print(f"Probabilidad de que la proporción muestral esté entre 0.55 y 0.65: {prob_entre_055_065:.4f}")

# Visualización
plt.figure(figsize=(10, 6))

# Histograma de la distribución muestral simulada
sns.histplot(proporciones_muestrales, kde=True, color='blue', bins=30, stat='density', label='Distribución Muestral Simulada')

# Distribución normal teórica
x = np.linspace(p - 4 * error_estandar, p + 4 * error_estandar, 1000)
y = norm.pdf(x, p, error_estandar)
plt.plot(x, y, 'r-', label='Distribución Normal Teórica')

# Sombrear región para P(𝑝̂ > 0.65)
x_fill1 = np.linspace(0.65, p + 4 * error_estandar, 1000)
y_fill1 = norm.pdf(x_fill1, p, error_estandar)
plt.fill_between(x_fill1, y_fill1, color='blue', alpha=0.3, label='P(𝑝̂ > 0.65)')

# Sombrear región para P(0.55 < 𝑝̂ < 0.65)
x_fill2 = np.linspace(0.55, 0.65, 1000)
y_fill2 = norm.pdf(x_fill2, p, error_estandar)
plt.fill_between(x_fill2, y_fill2, color='green', alpha=0.3, label='P(0.55 < 𝑝̂ < 0.65)')

plt.axvline(p, color='red', linestyle='--', label='Proporción Poblacional (0.6)')
plt.title('Distribución Muestral de Proporciones (n=100)')
plt.xlabel('Proporción Muestral (𝑝̂)')
plt.ylabel('Densidad')
plt.legend()
plt.show()