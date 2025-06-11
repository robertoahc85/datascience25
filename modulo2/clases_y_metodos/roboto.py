class Robot:
    def __init__(self, nombre, modelo):
        self.nombre = nombre # publica
        self._modelo = modelo # protegido
        self.__energia = 100 # privada 
        
    def obtener_modelo(self):
        return self._modelo
    
    def obtener_energia(self):
        return self.__energia
    
    def mostrar_info(self):
        print(f"Robot: {self.nombre}, Modelo: {self._modelo}, Energía: {self.__energia}")

robot1 = Robot("Robo1", "XJ-9")
print(robot1.nombre)  # Acceso público
print(robot1.obtener_modelo())  # Acceso protegido a través de un método  
print(robot1.obtener_energia())  # Acceso privado a través de un método  
robot1.mostrar_info()
robot1.nombre = "RoboModificado"  # Modificación de atributo público
robot1.mostrar_info()
robot1._modelo="Z-77"
robot1.mostrar_info()
robot1.__energia = 50  # Intento de modificación de atributo privado (no recomendado)
robot1.mostrar_info()
# print(robot1._modelo)  # Acceso protegido (no recomendado)
# print(robot1.__energia)  # Acceso protegido (no recomendado)