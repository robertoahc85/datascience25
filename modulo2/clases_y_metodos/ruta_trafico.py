import statistics
class Traffic_route:
    def __init__(self,route_id):
        self.route_id = route_id.upper()# Estandardizar el ID de la ruta a mayúsculas
        self.intersections = {} #Diccionario de intersecciones
        
    def add_intersection(self, name, time):
        #anade una intersección a la ruta
        if time >0: #verifica que el tiempo sea positivo
            self.intersections[name] = time  
        else:
            print("El tiempo debe ser positivo") 
              
    def total_time(self):
        #retorna el tiempo total de la ruta
        return sum(self.intersections.values())   
    
    def average_time(self, routes):
        #retorna el tiempo promedio de la ruta
        if not routes:
            return 0
        total_time = [route.total_time() for route in routes]
        return statistics.mean(total_time)
 
# def main():
    # Crear una ruta de tráfico
route_id = input("Ingrese el ID de la ruta: ")
route = Traffic_route(route_id)  # Estandarizar el ID de la ruta a mayúsculas
print("---------Creador de rutas de tráfico------")
while True:
    print("Muestra el menu de opciones:")
    print("1.agregar intersección")
    print("2.mostrar resultados")
    print("3.promedio de rutas")
    print("4.salir")
    opcion = input("Seleccione una opción: ")
    if opcion == "1":
        #Agregar intersección
        intersetcion_name = input("Ingrese el nombre de la intersección: ")
        intersetcion_time = int(input("Ingrese el tiempo de la intersección (en minutos): "))
        route.add_intersection(intersetcion_name, intersetcion_time)
    elif opcion == "2":
        #Mostrar resultados
        print(f"Tiempo total de {route.route_id}: {route.total_time()} minutos") 
    elif opcion == "3":
        #Calcular el tiempo promedio de las rutas
        average_time = route.average_time([route])
        print(f"Tiempo promedio de las rutas: {average_time} minutos")                               
    elif opcion == "4":
       #Salir del programa
        print("Saliendo del programa...")
        break      
    else:
        print("Opción no válida. Intente de nuevo.")
        
    
    # # while True:
    # #     ruta_id = input("Ingrese el ID de la ruta (o 'salir' para terminar): ")
    # #     if ruta_id.lower() == 'salir':  # Permitir al usuario salir del bucle
    # #         break    
    # #     route1 = Traffic_route(ruta_id)   
    # # route1 = Traffic_route("-------------RUTA1---------------")
    # # route1.add_intersection("Intersección A", 10)
    # # route1.add_intersection("Intersección B", 15)
    
    # # route2 = Traffic_route("RUTA2")
    # # route2.add_intersection("Intersección C", 20)
    # # route2.add_intersection("Intersección D", 25)
    
    # # Calcular el tiempo total de la ruta
    # print(f"Tiempo total de {route1.route_id}: {route1.total_time()} minutos")
    
    # # Calcular el tiempo promedio de las rutas
    # average_time = route1.average_time([route1, route2])
    # print(f"Tiempo promedio de las rutas: {average_time} minutos")
    
# if __name__ == "__main__":
#     main()