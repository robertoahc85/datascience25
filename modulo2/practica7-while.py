# contador = 1
# while contador <= 5:
#     print(f"Hola{contador}")
#     contador += 1

#var ir contando pero va ir validando si par o impar
# numero = 1
# while  numero <= 5:
#     if numero % 2 == 0:
#         print(f"El numero {numero} es par")
#     else:
#         print(f"El numero {numero} es impar")
#     numero += 1   

# Escribe un programa en Python que recorra cada letra 
# de una palabra utilizando un ciclo while. 
# El programa debe imprimir la posición de cada letra
# y la letra correspondiente. 
# Por ejemplo, si la palabra es "gato", el programa debe mostrar:

# Letra en posición 0: g  
# Letra en posición 1: a  
# Letra en posición 2: t  
# Letra en posición 3: o

# palabra= input("Ingrese una palabra:")
# palabra2 = palabra.lower()
# indice = 0
# while indice < len(palabra2):
#     print(f"Letra en posición {indice}: {palabra2[indice]}")
#     indice += 1

#Menu while 
opcion = ""
while opcion != "4":
    print("1. suma")
    print("2. resta 2")
    print("3. multiplicacion")
    print("4. Salir")
    opcion = input("Seleccione una opción: ")
    
    if opcion == "1":
        numero1 = float(input("Ingrese el primer número: "))
        numero2 = float(input("Ingrese el segundo número: "))
        resultado = numero1 + numero2
        print(f"La suma de {numero1} y {numero2} es: {resultado}")
    elif opcion == "2":
        numero1 = float(input("Ingrese el primer número: "))
        numero2 = float(input("Ingrese el segundo número: "))
        resultado = numero1 - numero2
        print(f"La resta de {numero1} y {numero2} es: {resultado}")
    elif opcion == "3":
        numero1 = float(input("Ingrese el primer número: "))
        numero2 = float(input("Ingrese el segundo número: "))
        resultado = numero1 * numero2
        print(f"La multiplicación de {numero1} y {numero2} es: {resultado}")
    elif opcion == "4":
        print("Saliendo del programa...")
    else:
        print("Opción no válida, por favor intente de nuevo.")
# Desarrolla un programa en Python que permita al usuario ingresar  4 números
#realizar operaciones aritméticas con ellos. 
# Usa un menú interactivo con las siguientes opciones:


# Calcular y mostrar la suma total de los números ingresados.

# Calcular y mostrar el promedio de los números.

# Mostrar el número mayor y menor de la lista.

# Salir del programa.

# El menú debe repetirse usando un ciclo while 
# hasta que el usuario elija salir. 
# Asegúrate de validar las entradas numéricas


