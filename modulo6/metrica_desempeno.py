# Predicción de precios de viviendas (House Prices Dataset):
# A partir de características de una propiedad (área habitable, 
# número de habitaciones, tamaño del terreno), se entrena un modelo de regresión lineal
# para estimar el precio de venta. Se evalúa el rendimiento del modelo mediante
# métricas estándar como MAE, MSE, RMSE y R², 
# buscando determinar qué tan precisa es la predicción de valores continuos.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------
# Teoría: Carga de Datos desde Kaggle
# ---------------------------------------------
"""
Teoría: Kaggle es una plataforma gratuita que ofrece datasets públicos descargables como archivos CSV.
Esta función carga datos desde un archivo local, evitando la necesidad de autenticación o servicios en la nube.
Facilita el acceso offline a datasets como 'House Prices' y 'Adult Census Income', promoviendo flexibilidad en el análisis.
Instrucciones: Descargue los archivos CSV desde las URLs proporcionadas, regístrese en Kaggle (gratis), y colóquelos en el directorio de trabajo.
"""
def cargar_datos_kaggle(ruta_archivo):
    """
    Carga un dataset CSV desde un archivo local descargado de Kaggle.
    """
    # Leer el archivo CSV en un DataFrame de Pandas
    data = pd.read_csv(ruta_archivo)
    
    # Retornar el DataFrame cargado para su uso posterior
    return data

# ---------------------------------------------
# Parte 1: Métricas de Regresión con Dataset House Prices
# ---------------------------------------------
"""
Teoría de Regresión: La regresión predice valores continuos basados en relaciones lineales o no lineales.
Aquí se usa Regresión Lineal, asumiendo una relación lineal entre características y el precio de venta.
Métricas:
- MAE: Promedio de errores absolutos, mide la magnitud media sin considerar dirección.
- MSE: Promedio de errores al cuadrado, penaliza más los errores grandes.
- RMSE: Raíz cuadrada de MSE, interpretable en las mismas unidades que el precio.
- R²: Proporción de varianza explicada, rango de 0 a 1 (1 indica ajuste perfecto).
Dataset: House Prices (URL: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
contiene datos de precios de casas en Ames, Iowa, para predecir el precio de venta.
"""
def ejemplo_regresion():
    # Cargar el dataset House Prices desde el archivo CSV descargado
    data_reg = cargar_datos_kaggle('input/train.csv')
    
    # Separar las características (X_reg) de la variable objetivo (y_reg: 'SalePrice' - precio de venta)
    X_reg = data_reg[['GrLivArea', 'BedroomAbvGr', 'LotArea']].dropna()  # Ejemplo de features relevantes
    y_reg = data_reg['SalePrice'].loc[X_reg.index]
    
    # Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%) para evaluar el modelo
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Inicializar un escalador para normalizar las características y mejorar el rendimiento del modelo
    scaler = StandardScaler()
    # Ajustar el escalador a los datos de entrenamiento y transformarlos
    X_train_reg_scaled = scaler.fit_transform(X_train_reg)
    # Transformar los datos de prueba usando el mismo escalador ajustado
    X_test_reg_scaled = scaler.transform(X_test_reg)
    
    # Inicializar el modelo de Regresión Lineal
    model_reg = LinearRegression()
    # Entrenar el modelo con los datos escalados de entrenamiento
    model_reg.fit(X_train_reg_scaled, y_train_reg)
    
    # Realizar predicciones en el conjunto de prueba
    y_pred_reg = model_reg.predict(X_test_reg_scaled)
    
    # Calcular el Error Absoluto Medio (MAE)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    # Calcular el Error Cuadrático Medio (MSE)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    # Calcular la Raíz del Error Cuadrático Medio (RMSE)
    rmse = np.sqrt(mse)
    # Calcular el Coeficiente de Determinación (R²)
    r2 = r2_score(y_test_reg, y_pred_reg)
    
    # Imprimir las métricas calculadas para evaluar el rendimiento del modelo
    print("\nMétricas de Regresión:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

# ---------------------------------------------
# Parte 2: Métricas de Clasificación con Dataset Adult Census Income
# ---------------------------------------------
"""
Teoría de Clasificación: La clasificación asigna etiquetas categóricas basadas en patrones aprendidos.
Aquí se usa Regresión Logística para clasificación binaria, modelando probabilidades con la función sigmoide.
Métricas:
- Matriz de Confusión: Tabla que resume predicciones correctas/incorrectas (TP, TN, FP, FN).
- Exactitud: Proporción de predicciones correctas totales.
- Precisión: Proporción de verdaderos positivos entre positivos predichos.
- Sensibilidad (Recall): Proporción de verdaderos positivos entre positivos reales.
- Especificidad: Proporción de verdaderos negativos entre negativos reales.
- ROC-AUC: Área bajo la curva ROC, mide la capacidad de discriminación (1 indica separación perfecta).
Dataset: Adult Census Income (URL: https://www.kaggle.com/datasets/uciml/adult-census-income)
para predecir si el ingreso excede 50K USD basado en datos demográficos.
"""
def ejemplo_clasificacion():
    # Cargar el dataset Adult Census Income desde el archivo CSV descargado
    data_clas = cargar_datos_kaggle('adult.csv')
    
    # Mapear la variable objetivo a binario: 1 si ingreso >50K, 0 otherwise
    data_clas['income'] = data_clas['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    # Aplicar one-hot encoding a variables categóricas para convertirlas en numéricas
    data_clas = pd.get_dummies(data_clas, drop_first=True)
    
    # Separar características (X_clas) de la variable objetivo (y_clas)
    X_clas = data_clas.drop('income', axis=1)
    y_clas = data_clas['income']
    
    # Dividir los datos en entrenamiento y prueba
    X_train_clas, X_test_clas, y_train_clas, y_test_clas = train_test_split(X_clas, y_clas, test_size=0.2, random_state=42)
    
    # Inicializar el escalador
    scaler = StandardScaler()
    # Ajustar y transformar datos de entrenamiento
    X_train_clas_scaled = scaler.fit_transform(X_train_clas)
    # Transformar datos de prueba
    X_test_clas_scaled = scaler.transform(X_test_clas)
    
    # Inicializar el modelo de Regresión Logística con límite de iteraciones
    model_clas = LogisticRegression(max_iter=200, random_state=42)
    # Entrenar el modelo
    model_clas.fit(X_train_clas_scaled, y_train_clas)
    
    # Realizar predicciones de clases
    y_pred_clas = model_clas.predict(X_test_clas_scaled)
    # Obtener probabilidades para la clase positiva (para ROC-AUC)
    y_prob_clas = model_clas.predict_proba(X_test_clas_scaled)[:, 1]
    
    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test_clas, y_pred_clas)
    # Imprimir la matriz de confusión
    print("\nMatriz de Confusión:")
    print(cm)
    
    # Calcular exactitud (accuracy)
    accuracy = accuracy_score(y_test_clas, y_pred_clas)
    # Calcular precisión
    precision = precision_score(y_test_clas, y_pred_clas)
    # Calcular sensibilidad (recall)
    recall = recall_score(y_test_clas, y_pred_clas)
    # Extraer TN, FP, FN, TP de la matriz para calcular especificidad
    tn, fp, fn, tp = cm.ravel()
    # Calcular especificidad
    specificity = tn / (tn + fp)
    # Calcular AUC de la curva ROC
    auc = roc_auc_score(y_test_clas, y_prob_clas)
    
    # Imprimir las métricas de clasificación
    print("\nMétricas de Clasificación:")
    print(f"Exactitud: {accuracy:.4f}")
    print(f"Precisión: {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")
    print(f"Especificidad: {specificity:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    
    # Crear figura para visualizar la matriz de confusión
    plt.figure(figsize=(6, 4))
    # Dibujar heatmap de la matriz
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    # Establecer título
    plt.title('Matriz de Confusión - Clasificación')
    # Etiquetar ejes
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    # Mostrar la gráfica
    plt.show()
    
    # Calcular tasas para la curva ROC
    fpr, tpr, _ = roc_curve(y_test_clas, y_prob_clas)
    # Crear figura para la curva ROC
    plt.figure(figsize=(6, 4))
    # Dibujar la curva ROC
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    # Dibujar línea diagonal de referencia
    plt.plot([0, 1], [0, 1], 'k--')
    # Establecer título
    plt.title('Curva ROC')
    # Etiquetar ejes
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    # Añadir leyenda
    plt.legend()
    # Mostrar la gráfica
    plt.show()

# ---------------------------------------------
# Función Principal
# ---------------------------------------------
"""
Teoría: La función principal orquesta la ejecución de los ejemplos de regresión y clasificación.
Esto permite una estructura modular, donde cada parte se ejecuta secuencialmente para demostrar las métricas.
"""
def main():
    """
    Ejecuta los ejemplos de métricas para regresión y clasificación.
    """
    # Imprimir encabezado y ejecutar el ejemplo de regresión
    print("Ejemplo de Regresión con House Prices Dataset:")
    ejemplo_regresion()
    
    # Imprimir encabezado y ejecutar el ejemplo de clasificación
    print("\nEjemplo de Clasificación con Adult Census Income Dataset:")
    ejemplo_clasificacion()

if __name__ == "__main__":
    # Ejecutar la función principal si el script se corre directamente
    main()