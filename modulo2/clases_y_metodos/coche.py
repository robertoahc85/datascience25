class Coche:
    def __init__(self, modelo, color,):
        self.marca = "Toyota"
        self.modelo = modelo
        self.color = color
        
    def arrancar(self):
        print(f"El coche {self.marca} {self.modelo} ha arrancado.")

    def detener(self):
        print(f"El coche {self.marca} {self.modelo} se ha detenido.")

    def mostrar_info(self):
        print(f"Coche: {self.marca} {self.modelo}, Color: {self.color}") 
  
#Vamos haer un lote de autos 
# marca= input("Dame la marca del coche: ")
modelo =input("Dame el modelo del coche: ")
color =input("Dame el color del coche: ")       
mi_coche = Coche(modelo, color)  
mi_coche.mostrar_info()   
mi_coche.arrancar()
mi_coche.detener()
el_coche = Coche("VW", "Citroneta", "Rojo")
el_coche.mostrar_info()  
            