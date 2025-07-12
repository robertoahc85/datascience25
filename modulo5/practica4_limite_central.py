# Escenario: Una población de tiempos de espera en una cafetería tiene una distribución exponencial 
# con una media de 6 minutos mu=6  y una desviación estándar de 6 minutos
# (, ya que en una distribución exponencial 6
# Se desea demostrar que la distribución muestral de la media para muestras de tamaño $  
# n = 40   es aproximadamente normal.

# Simular 1000 muestras de tamaño  n = 40  desde una distribución exponencial.
# Calcular la media muestral para cada muestra.
# Visualizar la distribución muestral de las medias y compararla con la distribución normal teórica.
# Calcular la media y el error estándar de las medias muestrales y compararlos con los valores teóricos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

#configurar semilla 
np.random.seed(123)

#Paramentros de la poblacion
mu = 6 #media poblacion minutos
sigma = 6 #Desviacion estandar poblacional(minutos)
n =40 #Tamano de la muestra
num_muestra = 1000

#Generar la medias muestrales
medias_muestrales = [np.mean(np.random.exponential(scale=mu, size=n)) for _ in range(num_muestra)]
#Calcular propiedades
media_dist_muestral = np.mean(medias_muestrales)
error_estandar_simulado = np.std(medias_muestrales, ddof=1)
error_estandar_teorico = sigma / np.sqrt(n)

#imprimir lo resultados 
print(f"Media de la distribuccion muestrales {media_dist_muestral} minutos ")
print(f"Error estadanr simulado {error_estandar_simulado} minutos ")
print(f"Error estadanr teorico {error_estandar_teorico} minutos ")

#Crear el dataframe
df_muestra = pd.DataFrame({'media_muestral': medias_muestrales})

#Visualizacion de los resultado
plt.figure(figsize=(10,6))
sns.histplot(df_muestra['media_muestral'],kde=True, color='blue', bins=30, stat='density', label='Distribuccion muestral simulada')
x = np.linspace(mu - 4 * error_estandar_teorico, mu + 4 * error_estandar_teorico, 1000 )
y= norm.pdf(x, mu, error_estandar_teorico)
plt.plot(x,y, 'r-', label = 'Distribucion normal teorica')
plt.axvline(media_dist_muestral, color='green', linestyle='--', label='Media muestral simulada')
plt.axvline(mu, color='black', linestyle=':', label='Media poblacional')
plt.xlabel('Media muestral')
plt.ylabel('Densidad')
plt.title('Distribución muestral de la media vs Normal teórica')
plt.legend()
plt.savefig('graficas/limite_central.png')
plt.show()