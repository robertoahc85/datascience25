import numpy as np
matriz = np.array([[5,10, 15], [20, 25, 30], [35, 40, 45]])  # Crea una matriz de 3x3
elemento = matriz[1, 2]  # Selecciona el elemento en la segunda fila y tercera columna
segunda_fila = matriz[1, :]  # Selecciona la segunda fila completa
print("Elemento en la segunda fila y tercera columna:", elemento)
print("Segunda fila completa:", segunda_fila)
mayor_que_20 = matriz[(matriz > 20) & (matriz < 40)]   # Selecciona todos los elementos mayores que 20
cantidad_mayores_que_20= mayor_que_20.size  # Cuenta la cantidad de elementos mayores que 20
cantidad = len(mayor_que_20)  # Otra forma de contar la cantidad de elementos mayores que 20
print("Elementos mayores que 20:", mayor_que_20)
print("Cantidad de elementos mayores que 20 pero menores que 40:", cantidad_mayores_que_20)
print("Cantidad de elementos mayores que 20 (otra forma):", cantidad)
# Selecciona todos los elementos de la primera fila