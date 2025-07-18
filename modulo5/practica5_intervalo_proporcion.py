# Ejercicio: Calcule un intervalo de confianza del 90% para 200
# personas con 60 a favor de una política.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

n= 200
p_hat = 60/n
z_critical = stats.norm.ppf(0.95)

#intervalo de confianza
margin_error = z_critical * np.sqrt(p_hat * (1- p_hat)/n)
ci = (p_hat - margin_error, p_hat + margin_error)
print(f"El intervalo de confianza del 90% para la proporción es: ({ci[0]:.3f}, {ci[1]:.3f})")

#Visualizacion 
# Simulamos datos binomiales para visualizar la proporción de apoyo
data = pd.DataFrame({'apoyo': np.random.binomial(1, p_hat, n)})

# Visualizamos la proporción de apoyo con un histograma
sns.histplot(data['apoyo'], bins=2, discrete=True)
plt.axvline(ci[0], color='red', linestyle='--', label=f'Límite inferior IC 90% {ci[0]}')
plt.axvline(ci[1], color='green', linestyle='--', label=f'Límite superior IC 90%{ci[1]}')
plt.legend()
plt.title('Histograma de Apoyo')
plt.xlabel('Apoyo (1=Sí, 0=No)')
plt.ylabel('Frecuencia')
plt.savefig('graficas/histograma_apoyo.png')
plt.show()

# Graficamos la proporción de apoyo
sns.countplot(x='apoyo', data=data)
plt.title('Distribución de Apoyo')
plt.xlabel('Apoyo (1=Sí, 0=No)')
plt.ylabel('Cantidad')
plt.savefig('graficas/intervalo_proporcion.png')
plt.show()
