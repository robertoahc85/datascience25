class Producto:
    def __init__(self, nombre, precio, cantidad):
        self.nombre = nombre
        self.precio = precio
        self.cantidad = cantidad

    def calcular_total(self):
        return self.precio * self.cantidad

    def aplicar_descuento(self, porcentaje):
        self.precio -= self.precio * (porcentaje / 100)

# Ejemplo de uso de la clase Producto    
producto = Producto("Laptop", 1000, 2)
producto.aplicar_descuento(10)
print(f"Precio con descuento: {producto.precio}")