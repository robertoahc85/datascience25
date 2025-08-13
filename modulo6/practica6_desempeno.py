#Problema: En el mercado inmobiliario, predecir el precio mediano de las casas en áreas específicas 
# es crucial para inversores, compradores y políticas urbanas. Sin embargo, factores como ingresos medianos, 
# población y ubicación influyen en los precios,haciendo difícil una predicción precisa sin modelos cuantitativos.
# El dataset 'California Housing Prices'
# (URL: https://www.kaggle.com/datasets/camnugent/california-housing-prices) 
# contiene datos de censo de 1990 sobre bloques de viviendas en California, con variables como longitud, latitud, 
# edad mediana de las casas, ingresos medianos y precio mediano de las casas.
# El objetivo es construir un modelo de regresión para predecir el precio mediano ('median_house_value')
# basándose en estas características, evaluando su precisión para identificar áreas subvaloradas o sobrevaloradas.
# Posible Solución: Usar Regresión Lineal para modelar la relación lineal entre características y precio.
# Escalar datos para mejorar el rendimiento. Evaluar con MAE (error promedio), MSE (penaliza errores grandes),
# RMSE (error en unidades de precio) y R² (variabilidad explicada).

#importar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score   
import matplotlib.pyplot as plt
import seaborn as sns   
#cargar el dataset  
def cargar_dato(ruta_archivo):
    data = pd.read_csv(ruta_archivo)
    #Separa las características (X) y la variable objetivo(y)
    X = data.drop('median_house_value', axis=1).select_dtypes(include=[np.number])#Usar solo columnas numéricas
    y = data['median_house_value']
    return X, y
#Dividir el dataset en conjunto de entrenamiento y prueba
def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
#Preparacion y escalado de los datos
def preparar_datos(X_train, X_test):
    X_train = X_train.fillna(X_train.mean())  # Rellenar valores faltantes con la media
    X_test = X_test.fillna(X_test.mean())    # Rellenar valores faltantes con la media
    scaler = StandardScaler()  # Inicializar el escalador
    X_train_scaled = scaler.fit_transform(X_train)  # Ajustar y transformar los datos
    X_test_scaled = scaler.transform(X_test)  # Transformar los datos de prueba
    return X_train_scaled, X_test_scaled

def entrenar_modelo(X_train_scaled, y_train):
    modelo = LinearRegression()  # Inicializar el modelo de regresión lineal
    modelo.fit(X_train_scaled, y_train)  # Entrenar el modelo con los datos escalados
    return modelo

def evaluar_modelo(modelo, X_test_scaled, y_test):
    y_pred = modelo.predict(X_test_scaled)  # Predecir los valores de prueba
    mae = mean_absolute_error(y_test, y_pred)  # Calcular el error absoluto medio
    mse = mean_squared_error(y_test, y_pred)  # Calcular el error cuadrático medio
    rmse = np.sqrt(mse)  # Calcular la raíz del error cuadrático medio
    r2 = r2_score(y_test, y_pred)  # Calcular el coeficiente de determinación R²
     # Imprimir las métricas de evaluación
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")
    
    #Visualizar los resultados
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)      
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Línea de referencia
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales')
    plt.show()
#Función principal para ejecutar el flujo de trabajo
def main():
    ruta_archivo = 'input/housing.csv'  # Ruta del archivo CSV
    X, y = cargar_dato(ruta_archivo)  # Cargar los datos
    X_train, X_test, y_train, y_test = dividir_datos(X, y)  # Dividir los datos
    X_train_scaled, X_test_scaled = preparar_datos(X_train, X_test)  # Preparar y escalar los datos
    modelo = entrenar_modelo(X_train_scaled, y_train)  # Entrenar el modelo
    evaluar_modelo(modelo, X_test_scaled, y_test)  # Evaluar el modelo
if __name__ == "__main__":
    main()  # Ejecutar la función principal
#Este código implementa un modelo de regresión lineal para predecir el precio med