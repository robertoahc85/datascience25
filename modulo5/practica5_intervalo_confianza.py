# Ejercicio: Calcule un intervalo de confianza del 99% para 60
# cables con media 500 MPa y σ = 15 MPa.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
#Datos
n=60
mean_sample = 500
std_pop = 15
z_critical = stats.norm.ppf(0.995)
#intervalo de confianza
margin_error = z_critical * (std_pop / np.sqrt(n))
ci = (mean_sample - margin_error, mean_sample + margin_error)
print(f"Intervalo de confianza 99%: {ci}")

#visualizacion
data = pd.DataFrame({'resistencia': np.random.normal(mean_sample,std_pop,n)})
sns.histplot(data['resistencia'], kde=True)
plt.axvline(ci[0], color='red', linestyle='--', label=f'Límite inferior IC lower: {ci[0]:.2f}')
plt.axvline(ci[1], color='green', linestyle='--', label=f'Límite superior IC lower"{ci[1]:.2f}')
plt.title('Distribución de resistencia de los cables')
plt.xlabel('Resistencia (MPa)')
plt.legend()
plt.savefig('graficas/intervaloconfianza.png')
plt.show()


