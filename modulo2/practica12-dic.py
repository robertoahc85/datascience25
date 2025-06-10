# persona = {
#     "nombre": "Juan",
#     "edad": 30,
#     "ciudad": "Madrid",
# }
# print("Nombre:", persona["nombre"])
# print("Edad:", persona["edad"])
# print("Ciudad:", persona["ciudad"])

usuarios = {
    "usuario1": {
        "nombre": "Juan",
        "edad": 30,
        "ciudad": "Madrid",
    },
    "usuario2": {
        "nombre": "Ana",
        "edad": 25,
        "ciudad": "Barcelona",
    },
    "usuario3": {
        "nombre": "Luis",
        "edad": 35,
        "ciudad": "Valencia",
    },
}

edades = [datos["edad"] for datos in usuarios.values()]
promedio_edad = sum(edades) / len(edades)
print("Promedio de edad de los usuarios:", promedio_edad)