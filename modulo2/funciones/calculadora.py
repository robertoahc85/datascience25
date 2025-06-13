import math
import statistics
def suma(a, b):
    """Suma dos números."""
    return a + b
def resta(a, b):
    """Resta dos números."""
    return a - b 
def multiplicacion(a, b):
    """Multiplica dos números."""
    return a * b
def division(a, b):
    """Divide dos números."""
    if b == 0:
        raise ValueError("No se puede dividir por cero.")
    return a / b   

numero1 = 10
numero2 = 5
suma = suma(numero1, numero2)
resta = resta(numero1, numero2)
multiplicacion = multiplicacion(numero1, numero2)
division = division(numero1, numero2)   
raiz_cuadrada = math.sqrt(numero1)   
datos = [numero1, numero2, suma, resta, multiplicacion, division, raiz_cuadrada]
media = statistics.mean(datos)  
pi = math.pi
median = statistics.median(datos)
logaritmo = math.log(numero1)  # Logaritmo natural de numero1
seno = math.sin(numero1)  # Seno de numero1
coseno = math.cos(numero1)  # Coseno de numero1
maximos = max(datos)  # Máximo de los datos
minimos = min(datos)  # Mínimo de los datos
potencia = math.pow(numero1, 2)  # Potencia de numero1 al cuadrado
print(f"Valor de pi: {pi}")
print(f"Suma: {suma}")
print(f"Resta: {resta}") 
print(f"Multiplicación: {multiplicacion}")
print(f"División: {division}")
print(f"Raíz cuadrada de {numero1}: {raiz_cuadrada}")
print(f"Datos: {datos}")   
print(f"Media de los datos: {media}")
    
       