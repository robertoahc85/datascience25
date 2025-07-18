# Ejercicio: Estime la proporción de defectos en una fábrica con margen de error ±0.03 y
# 95% de confianza, sin datos previos. - Solución Manual:
import numpy as np
import scipy.stats as stats

# Datos
margin_error = 0.03
z_critical = stats.norm.ppf(0.975)
p = 0.5

#Tamano muetral
n = (z_critical**2  * p * (1 - p)) / (margin_error**2)
n = np.ceil(n)
print(f"El tamaño de muestra necesario es: {int(n)}")

