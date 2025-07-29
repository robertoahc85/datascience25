#Regresion Lineal
# Definición del Problema: Predecir un valor numérico continuo, como el precio
# de una casa, a partir de características como su tamaño.

import numpy as np
import matplotlib.pyplot  as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Establecer semilla
np.random.seed(42)
X = 2 * np.random.rand(100,1)
y = 4 + 3 * X + np.random.rand(100,1)

#Dividir datos en 80% para entrenamiento, 20% para prueba
X_train ,X_test ,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Crear y entrenar el modelo regresion lineal
model = LinearRegression()
model.fit(X_train,y_train)

#Predicir el conjuto de pruebas
y_pred = model.predict(X_test)

#Calcular el error cuadractico medio
mse = mean_squared_error(y_test, y_pred)
print(f"Means Squared Error : {mse:2f}")

#Visualizar datos reales y la recta predicha
plt.scatter(X_test,y_test, color='blue', label="Datos reales")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Recta predicha")
plt.xlabel("Tamano de lacasa ")
plt.ylabel("Precios")
plt.title("Regresión Lineal")
plt.legend()
plt.savefig("graph/regresion_lineal.png")
plt.show()
print(f"Coeficiente(pendiente){model.coef_[0][0]}")
print(f"interceptcion(bias){model.intercept_[0]}")
