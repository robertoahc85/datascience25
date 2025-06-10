# #programa que validas si un numero es par o impar
numero = int(input("Ingrese un numero: "))
if numero % 2 == 0:
    print("El numero es par") 
else:
    print("El numero es impar")    
    
paridad = "par" if numero % 2 == 0 else "impar"    
print(f"Estes un numero {paridad} el numero es {numero}")
print(paridad)

    
# #programa que valida si un numero es positivo o negativo    
# # numero = int(input("Ingrese un numero: "))
# if numero >= 0:
#     print("El numero es positivo")  
# else:
#     print("El numero es negativo")      
# #Verifica un usuario y contraseña
# usuario = input("Ingrese su usuario: ")
# contrasena = input("Ingrese su contraseña: ")    
# if usuario == "admin" and contrasena == "1234":
#     print("Acceso concedido") 
# else:
#     print("Acceso denegado")    
#  que valide la capital de un pais 
# capital = input("Cual es la capital de Chile ")
# if capital.upper == "SANTIAGO":
#     print("Bien hecho, la capital de Chile es Santiago")
# else:
#     print("Lo siento, la capital de Chile no es " + capital + ", es Santiago")
#Asignar una letra segun la calificacion
calificacion = int(input("Ingrese su calificacion: "))
if calificacion > 100 or calificacion < 0:
    print("Calificacion invalida")
else:
    if calificacion >= 90:
        print("A")
    elif calificacion >= 80:
        print("B")
    elif calificacion >= 70:
        print("C")   
    elif calificacion >= 60:
        print("D")  
    else:
        print("F")   
#Crea un programa en Python que pida al usuario su edad y
# luego muestre una clasificación según el rango de edad ingresado.
# Las categorías son las siguientes:
# Menos de 0: "Edad inválida"
# De 0 a 12 años: "Niño"
# De 13 a 17 años: "Adolescente"
# De 18 a 59 años: "Adulto"
# 60 años o más: "Adulto mayor"
#que no acepte edades negativas ni mayores a 120

  