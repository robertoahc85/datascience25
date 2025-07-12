# Escenario: Una población de tiempos de respuesta de un servidor tiene una media de 200 ms
# y una distribución sesgada (exponencial). 
# Muestra cómo la media muestral converge a 200 ms para tamaños n = 10, 100, 1000 .



import numpy as np
import matplotlib.pyplot as plt
#configurar semilla
np.random.seed(123)
#Parametros
mu = 200  # media de la población
tamanos_muestra = [10, 100, 1000]  # tamaños de muestra

medias_por_tamano = []
for n in tamanos_muestra:
    medias_muestrales = [np.mean(np.random.exponential(scale=mu, size=n)) for _ in range(1000)]
    medias_por_tamano.append(medias_muestrales)
    

# Visualizar resultados    
plt.figure(figsize=(12, 6))
plt.boxplot([medias_por_tamano[n] for n in range(len(tamanos_muestra))], label="Media poblacional")
plt.axhline(mu, color='red', linestyle='--', label='Media poblacional (200 ms)')
plt.title('Ley de los grande numeros: Convergencia de la media poblacional')
plt.xlabel('Tamaño de la muestra (n)')
plt.ylabel('Media muestral')
plt.legend()
plt.grid()
plt.show()
plt.savefig('graficas/ley_grande_numero.png')

#imprimir los resultados
for n in tamanos_muestra:   
    print(f'Tamaño de muestra: {n}, Media muestral promedio: {np.mean(medias_por_tamano[tamanos_muestra.index(n)])}, Desviación estándar: {np.std(medias_por_tamano[tamanos_muestra.index(n)])}')
