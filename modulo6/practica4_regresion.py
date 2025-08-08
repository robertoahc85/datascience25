#Practica " Regresion  con Dataset real sckit-learn"
#Dataset: load_diabetes

#objetivo: Comprender y aplicar modelos de regresion lineal con dataset real

#Importar librerias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split        
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score         
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

# Cargar el dataset
diabetes = load_diabetes()
X = diabetes.data #Variables independientes
y = diabetes.target #Variable dependiente
#Convertir a DataFrame para mejor visualización
df = pd.DataFrame(data=X, columns=diabetes.feature_names)
print("Dataset original:")
print(df.head())
print("Informarción general del dataset:")
print(df.info())
print("Estadisticas descriptivas:")
print(df.describe())

# Dividir el dataset en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   

# Aplicar  diferentes modelos de regresión
# Modelo de regresión lineal
modelo_lineal = LinearRegression()
modelo_lineal.fit(X_train, y_train)
y_pred_lineal = modelo_lineal.predict(X_test)

#Modelo2: Ridge Regression(penaliza coeficientes grandes) 
#Reduce  el sobreajuste
#multicolinealidad
modelo_ridge = Ridge(alpha=1.0)
modelo_ridge.fit(X_train, y_train)
y_pred_ridge = modelo_ridge.predict(X_test)

#Modelo3: Regresión Polinomica (grado 2)

modelo_poly = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),    # genera variables polinómicas
    ('scaler', StandardScaler()),              # escala los datos
    ('linear', LinearRegression())             # modelo final
])

modelo_poly.fit(X_train, y_train)
y_pred_poly = modelo_poly.predict(X_test)

#Modelo4: Lasso Regression (penaliza coeficientes grandes) (Leat Absolute Shrinkage and Selection Operator)
# Reduce el sobreajuste y la multicolinealidad
#La regularizacion L1 anade penalizacion proporcional a la suma de valor  absoluto de los coeficientes
modelo_lasso = Lasso(alpha=1.0)
modelo_lasso.fit(X_train, y_train)
y_pred_lasso = modelo_lasso.predict(X_test)

# Comparar  modelos y selececionar el mejor
def calcular_metricas(y_true, y_pred):
    """Calcula y retorna las métricas de evaluación: MSE y R²"""
    return{
        'MSE': mean_squared_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred) ,
        'MAE': mean_absolute_error(y_true, y_pred)    
     }

# Crear tabla con metricas de cada modelo
from sklearn.linear_model import Lasso

#Modelo4: Lasso Regression
modelo_lasso = Lasso(alpha=1.0)
modelo_lasso.fit(X_train, y_train)
y_pred_lasso = modelo_lasso.predict(X_test)

resultados = pd.DataFrame([
    {'Modelo': "Lineal", **calcular_metricas(y_test, y_pred_lineal)},  
    {'Modelo': "Ridge", **calcular_metricas(y_test, y_pred_ridge)},
    {'Modelo': "Polinomial", **calcular_metricas(y_test, y_pred_poly)},
    {'Modelo': "Lasso", **calcular_metricas(y_test, y_pred_lasso)}     
]).round(4)

#Visualizar resultados
plt.figure(figsize=(10, 6))
plt.bar(resultados['Modelo'], resultados['R2'], color=['blue', 'orange', 'green', 'red'])
plt.title('Comparación de Modelos de Regresión')
plt.xlabel('Modelo')
plt.ylabel('R²')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.savefig('graph/comparacion_modelos_regresion.png')
plt.show()  

#Mejor modelo
mejor_modelo = resultados.loc[resultados['R2'].idxmax()]


#Genera un dashboard con las metricas de los modelos

