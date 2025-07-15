import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
# Datos simulados de alturas
np.random.seed(42)
data = pd.DataFrame({'altura':np.random.normal(170,10,500)})
sample_mean = data['altura'].mean()
print(f"Media muestra:{sample_mean:.2f}cm")
#calculo de  erro standar
sample_std = data ["altura"].std(ddof=1)
n = len(data)
error_estandard = sample_std / np.sqrt(n)
#intervalo de confianza del 95 normal
z= norm.ppf(0.975)
ic_inf = sample_mean - z * error_estandard
ic_sup = sample_mean + z * error_estandard
#Visualizacion
plt.hist(data['altura'],bins=30, color="skyblue",edgecolor="black")
plt.axvline(sample_mean,color='red', linestyle='dashed',linewidth=2, label=f"Media ={sample_mean:.2f}")
#linea de intervalo de confianza
plt.axvline(ic_inf,color='green', linestyle="--", linewidth=2 , label =f"IC 95%")
plt.axvline(ic_sup,color='green', linestyle="--", linewidth=2 )
plt.title("Distribución de alturas simuladas")
plt.xlabel("Altura (cm)")
plt.ylabel("Frecuencia")
plt.savefig("graficas/inferencia.png")
plt.show()
