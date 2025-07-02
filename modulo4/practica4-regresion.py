#1 importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import mean_squared_error ,mean_absolute_error,r2_score

# Carga el cojunto de datos
df = pd.read_csv("entrada/forest_growth_data.csv")
print("Primeras filas dataset")
print(df.head(5))

#3. Visualizar  los datos
# Queremos ver la relacion entre tree_age y tree_heigth

plt.scatter(df['tree_age'],df["tree_height"],alpha=0.7)
plt.title("Edad del Arbol vs Altura")
plt.xlabel("Edad del Arbol (años)")
plt.ylabel("Altura del arbol (m)")
plt.grid(True)


# 4. Ajustar el modelo de regresion lineal Simple
#usamos  tree_age y tree_heigth
X = df[['tree_age']]#matriz (n,1)
y = df['tree_height']#vector (n,)
model = LinearRegression()
model.fit(X, y)

#5.Ver el resumen del modelo.
print("Resumen del Modelo")
print(f"Intercepto (β0):{model.intercept_:.3f}")
print(f"Pendiente (β1) {model.coef_[0]:.3f}")

#6 Intepretacion del Modelo
print(f"Intepretacion por cada año adicional de edad, la altura promedio del arbol aumenta {model.coef_[0]:.2f}metros")

#7. Visualizar la linea de Regresion
y_pred = model.predict(X)
plt.plot(df['tree_age'], y_pred, color='red', label="Linea de Regresion")
plt.title("Regresion lineal Simple")
plt.xlabel("Edad del arbol (años)")
plt.ylabel("Altura del arbol(m)")
plt.legend()
plt.grid(True)
plt.show()

#8 Evaluar el rendimiento  del modelo 
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\Evaluacion del Modelo")
print(f"Error Cuadratico Medio (MSE): {mse:.2f}")
print(f"Erro Absoluto Medio (MAE){mae:.2f}")
print(f"coeficiente de Determinacion (R2):{r2:.3f} ")