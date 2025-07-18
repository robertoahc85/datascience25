import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import pandas as pd

# Calcule intervalos del 90%, 95% y 99% para 100 valores
# con media 50 y s = 5.

#Datos
n = 100
mean_sample =50
std_sample = 5
df = n -1
conf_level = [0.90,0.95,0.99]
cis= {}

#intervalo de confianza
for conf in conf_level:
    t_critical = stats.t.ppf((1+conf)/2,df)
    margin = t_critical * (std_sample/np.sqrt(n))
    cis[conf] = (mean_sample -margin, mean_sample + margin)
print("intervalo de confianza",cis)    

#visualizacion
color_map = {
    0.90: 'blue',
    0.95: 'orange',
    0.99: 'purple' 
}

data = pd.DataFrame({'valor': np.random.normal(mean_sample,std_sample,n)})
plt.hist(data['valor'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
for conf, (lower, upper) in cis.items():
    color = color_map.get(conf,'gray')
    plt.axvline(lower, color=color, alpha=0.5,linestyle='--', label=f'{int(conf*100)}% IC lower')
    plt.axvline(upper, color=color, linestyle='--', label=f'{int(conf*100)}% IC upper', alpha=0.5)
plt.xlabel('Valor')
plt.ylabel('Frecuencia')
plt.title('Histograma con Intervalos de Confianza')
plt.legend()
plt.savefig('graficas/nivel_confianza.png')
plt.show()