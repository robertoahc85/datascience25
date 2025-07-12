import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parámetros de la población y la muestra
mu = 150  # Media poblacional (gramos)
sigma = 20  # Desviación estándar poblacional (gramos)
n = 36  # Tamaño de la muestra

# Calcular el error estándar
error_estandar = sigma / np.sqrt(n)

# 1. Probabilidad de que la media muestral sea mayor a 155 gramos
x1 = 155
z1 = (x1 - mu) / error_estandar
prob_mayor_155 = 1 - norm.cdf(z1)

# 2. Probabilidad de que la media muestral esté entre 148 y 152 gramos
x2_lower = 148
x2_upper = 152
z2_lower = (x2_lower - mu) / error_estandar
z2_upper = (x2_upper - mu) / error_estandar
prob_entre_148_152 = norm.cdf(z2_upper) - norm.cdf(z2_lower)

# Imprimir resultados
print(f"Error estándar: {error_estandar:.2f} gramos")
print(f"Probabilidad de que la media muestral sea mayor a 155 gramos: {prob_mayor_155:.4f}")
print(f"Probabilidad de que la media muestral esté entre 148 y 152 gramos: {prob_entre_148_152:.4f}")

# Visualización de la distribución muestral
x = np.linspace(mu - 4 * error_estandar, mu + 4 * error_estandar, 1000)
y = norm.pdf(x, mu, error_estandar)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', label='Distribución Muestral de la Media')
plt.axvline(mu, color='red', linestyle='--', label='Media Poblacional (150 g)')

# Sombrear región para P(𝜇 > 155)
x_fill1 = np.linspace(155, mu + 4 * error_estandar, 1000)
y_fill1 = norm.pdf(x_fill1, mu, error_estandar)
plt.fill_between(x_fill1, y_fill1, color='blue', alpha=0.3, label='P(𝜇 > 155)')

# Sombrear región para P(148 < 𝜇 < 152)
x_fill2 = np.linspace(148, 152, 1000)
y_fill2 = norm.pdf(x_fill2, mu, error_estandar)
plt.fill_between(x_fill2, y_fill2, color='green', alpha=0.3, label='P(148 < 𝜇 < 152)')

plt.title('Distribución Muestral de la Media (n=36)')
plt.xlabel('Media Muestral (gramos)')
plt.ylabel('Densidad')
plt.legend()
plt.show()