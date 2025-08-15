# Enunciado del problema
# El presente código busca resolver el problema del desbalance de clases en un conjunto
# de datos de detección de fraude con tarjetas de crédito, en el que las transacciones
# fraudulentas (clase minoritaria) son muy escasas en comparación con las transacciones
# legítimas (clase mayoritaria). Este desbalance provoca que los modelos de clasificación
# tiendan a predecir mayoritariamente la clase mayoritaria, reduciendo la capacidad de detectar 
# fraudes reales.

# El objetivo del código es evaluar el impacto de distintas técnicas 
# de balanceo de datos —submuestreo, sobremuestreo, SMOTE y SMOTEENN— aplicadas sobre el 
# conjunto de entrenamiento, y comparar el rendimiento de un modelo de Regresión Logística en cada escenario. Para ello:

# Se carga y divide el dataset manteniendo la proporción original de clases.

# Se aplican las diferentes estrategias de balanceo para generar conjuntos
# de datos más equilibrados.

# Se entrena el modelo con cada versión balanceada y se evalúa usando métricas 
# como precisión, recall, F1-score, matriz de confusión y ROC-AUC.

# Se visualizan los resultados mediante mapas de calor y curvas ROC para analizar 
# el efecto del balanceo en la detección de la clase minoritaria.

# En suma, el código aborda el desafío de mejorar la detección de casos de fraude 
# en un entorno de datos altamente desbalanceado mediante la comparación de técnicas
# de re-muestreo y la evaluación exhaustiva de su impacto en el desempeño del modelo.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------------------------------------
# Teoría: Importar las Librerías
# ---------------------------------------------
"""
Teoría: El primer paso es importar las librerías necesarias para el procesamiento de datos, modelado, evaluación y técnicas de balanceo.
Pandas se usa para manejar datos, scikit-learn para modelado y métricas, e imblearn para técnicas de balanceo como undersampling, oversampling, SMOTE y SMOTEENN.
Estas librerías permiten abordar problemas de desbalanceo en datasets, donde una clase es minoritaria (e.g., fraude en transacciones).
Dataset: Credit Card Fraud Detection (URL: https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Descargue 'creditcard.csv' y use 'Class' como objetivo (1 para fraude, altamente desbalanceado).
"""

# ---------------------------------------------
# Teoría: Carga los Datos
# ---------------------------------------------
"""
Teoría: Cargar los datos desde un archivo CSV local descargado de Kaggle.
Esto permite explorar el dataset y verificar el desbalanceo inicial de clases.
Instrucciones: Descargue el archivo desde la URL, colóquelo en el directorio de trabajo, y use pd.read_csv para cargarlo.
"""
def carga_datos(ruta_archivo):
    """
    Carga un dataset CSV desde un archivo local descargado de Kaggle.
    """
    # Separar características (X) y objetivo (y)
    X = data.drop('Class', axis=1)
    y = data['Class']
    
    # Retornar X y y para uso posterior
    return X, y

# ---------------------------------------------
# Teoría: División de Datos
# ---------------------------------------------
"""
Teoría: Dividir el dataset en entrenamiento y prueba para evaluar el modelo de manera imparcial.
Use train_test_split con estratificación para mantener el desbalanceo en ambos conjuntos.
Esto asegura que el modelo se entrene y evalúe en datos representativos.
"""
def dividir_datos(X, y):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    """
    # Dividir con estratificación para preservar desbalanceo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Retornar los conjuntos divididos
    return X_train, X_test, y_train, y_test

# ---------------------------------------------
# Teoría: Aplicación de las Técnicas de Balanceo
# ---------------------------------------------
"""
Teoría: Las técnicas de balanceo abordan el desbalanceo de clases:
- Submuestreo (Undersampling): Reduce la clase mayoritaria eliminando muestras aleatorias.
- Sobremuestreo (Oversampling): Aumenta la clase minoritaria duplicando muestras.
- Combinación de Submuestreo y Sobremuestreo (SMOTEENN): Usa SMOTE para oversampling y ENN para limpiar ruido.
- Generación de Muestras Sintéticas (SMOTE): Crea muestras sintéticas de la clase minoritaria interpolando vecinos.
Aplique cada técnica al conjunto de entrenamiento y compare el rendimiento del modelo.
"""
def aplicar_balanceo(X_train, y_train):
    """
    Aplica técnicas de balanceo y retorna los datos balanceados para cada método.
    """
    # Submuestreo (Undersampling)
    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
    
    # Sobremuestreo (Oversampling)
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)
    
    # Combinación de Submuestreo y Sobremuestreo (SMOTEENN)
    smoteenn = SMOTEENN(random_state=42)
    X_train_smoteenn, y_train_smoteenn = smoteenn.fit_resample(X_train, y_train)
    
    # Generación de Muestras Sintéticas (SMOTE)
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Retornar diccionario con datos balanceados
    return {
        'original': (X_train, y_train),
        'undersampling': (X_train_rus, y_train_rus),
        'oversampling': (X_train_ros, y_train_ros),
        'smoteenn': (X_train_smoteenn, y_train_smoteenn),
        'smote': (X_train_smote, y_train_smote)
    }

# ---------------------------------------------
# Teoría: Entrenamiento y Evaluación del Modelo
# ---------------------------------------------
"""
Teoría: Entrene un modelo (e.g., Regresión Logística) en los datos balanceados y evalúe con métricas como precisión, recall, F1-score, matriz de confusión y ROC-AUC.
Compare el modelo en datos originales vs. balanceados para ver mejoras en la detección de la clase minoritaria.
Use classification_report y confusion_matrix para evaluación detallada.
Visualice resultados con heatmaps y curvas ROC.
"""
def entrenar_evaluar(X_train, y_train, X_test, y_test, tecnica):
    """
    Entrena un modelo de Regresión Logística y evalúa su rendimiento.
    """
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Inicializar y entrenar modelo
    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predicciones y probabilidades
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Imprimir reporte de clasificación
    print(f"\nReporte de Clasificación ({tecnica}):")
    print(classification_report(y_test, y_pred))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {tecnica}')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.show()
    
    # ROC-AUC y curva
    auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f'Curva ROC - {tecnica}')
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.legend()
    plt.show()

# ---------------------------------------------
# Función Principal
# ---------------------------------------------
def main():
    """
    Ejecuta el ejercicio guiado para manejo de datos desbalanceados.
    """
    # Cargar datos
    data = pd.read_csv('creditcard.csv')
    
    # Verificar desbalanceo
    print("Distribución original de clases:", Counter(data['Class']))
    
    # Dividir datos
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Aplicar técnicas de balanceo y entrenar
    balanceados = aplicar_balanceo(X_train, y_train)
    
    # Entrenar y evaluar para cada técnica
    for tecnica, (X_bal, y_bal) in balanceados.items():
        print(f"\nEvaluando con {tecnica}:")
        entrenar_evaluar(X_bal, y_bal, X_test, y_test, tecnica)

if __name__ == "__main__":
    main()