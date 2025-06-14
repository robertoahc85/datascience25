import numpy as np
arreglo_arrange = np.arange(16, 40, 1)  # Crea un arreglo con valores desde 10 hasta 50 con un paso de 5
puntos = np.linspace(0, 10, 10)
print("arreglo_arrange:", arreglo_arrange)
print("puntos:", puntos)
#matriz cero
matriz_cero = np.zeros((4, 4)) # Crea una matriz de ceros de 4 filas y 4 columnas
matriz_uno = np.ones((4, 4)) # Crea una matriz de unos de 4 filas y 4 columnas
matriz_aletoria = np.random.rand(4, 4) # Crea una matriz aleatoria de 4 filas y 4 columnas
print("matriz_cero:\n", matriz_cero)
print("matriz_uno:\n", matriz_uno)          
print("matriz_aletoria:\n", matriz_aletoria)
