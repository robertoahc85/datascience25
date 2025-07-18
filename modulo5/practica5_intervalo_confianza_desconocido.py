# Calcule un intervalo de confianza del 95% para 30 rocas
# con media 10 kg y s = 1.5 kg.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

#Datos
n=30
mean_sample = 10
std_sample = 1.5
confianza= 0.95
df = n -1
# t_critical = stats.t.ppf(0.975, df)
t_critical = stats.t.ppf(1 - (1 - confianza) / 2, df)

#intervalo de confianza
margin_error = t_critical * (std_sample /np.sqrt(n))
ci = (mean_sample - margin_error, mean_sample + margin_error)
print(f"Intervalo de confianza del 95%: ({ci[0]:.2f}, {ci[1]:.2f})")

#Visualizacion
data = pd.DataFrame({'peso': np.random.normal(mean_sample,std_sample,n)})
sns.histplot(data=data, x='peso', kde=True)
plt.axvline(ci[0], color='red', linestyle='--', label=f'Limite inferior ({ci[0]:.2f})')
plt.axvline(ci[1], color='green', linestyle='--', label=f'Limite superior ({ci[1]:.2f})')
plt.axvline(mean_sample, color='blue', linestyle='-', label=f'Media ({mean_sample})')
plt.legend()
plt.title('Distribución de pesos de las rocas con intervalo de confianza')
plt.xlabel('Peso (kg)')
plt.ylabel('Frecuencia')
plt.savefig('graficas/desconocido.png')
plt.show()
