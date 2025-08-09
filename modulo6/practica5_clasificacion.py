import sqlite3
import pandas as pd
import numpy as np      
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


#----------
# Conexión a la base de datos y Carga de datos
#----------
def cargar_datos():
    iris = load_iris()
    # Crear un DataFrame de pandas  
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    #conectar una base de datos SQLite
    conn = sqlite3.connect('iris.db')
    # Guardar el DataFrame en la base de datos
    df.to_sql('iris_table', conn, if_exists='replace', index=False)  
    # Consultar los datos
    query = "SELECT * FROM iris_table"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

#Preparar los datos
def preparar_datos(data):
    data =cargar_datos
    #Separar las características 
    X = data.drop('target', axis=1)
    #Separar la variable objetivo
    y = data['target']
    return X, y

#Dividir los datos en conjuntos de entrenamiento y prueba
def dividir_datos(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 
 
 
def graficar_matriz_confusion(y_test, predicciones, nombre_modelo):
    #Calcular la matriz de confusión
    cm = confusion_matrix(y_test, predicciones)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Setosa','Versicolor', 'Virgnica'], yticklabels=['Setosa','Versicolor', 'Virgnica'])
    plt.xlabel('Predicted')
    plt.ylabel('Real')
    plt.title('Confusion Matrix - ' + nombre_modelo)
    plt.show()   
    
#Modelo de Regresión Logística
#Clasificación binaria o multiclase que modelaa la probabilidad de pertenencia a una clase
#utiliza funciona sigmoide para predecir la probabilidad de pertenencia a una clase
def entrenar_regresion_logistica(X_train, X_test, y_train, y_test):
    modelo = LogisticRegression(max_iter=200, random_state=42)
    #Entrenar el modelos con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    #Realizar predicciones de los datos de prueba
    predicciones = modelo.predict(X_test)
    #Graficar la matriz de confusión
    graficar_matriz_confusion(y_test, predicciones, 'Regresión Logística ')
    return modelo , predicciones, accuracy_score(y_test, predicciones)

#Modelo de arbol de decisión
#Clasificación  o regresion mediante particiones recursivas del espacio de características
#-Como funciona: divide los datos en regiones basadas en Reglas de decisiónes
#(e.g "Si x1 < 5.0, entonces clase A")
def entrenar_arbol_decision(X_train, X_test, y_train, y_test):
    modelo = DecisionTreeClassifier(random_state=42)
    #Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    #Realizar predicciones de los datos de prueba
    predicciones = modelo.predict(X_test)
    #Graficar la matriz de confusión
    graficar_matriz_confusion(y_test, predicciones, 'Árbol de Decisión')
    return modelo, predicciones, accuracy_score(y_test, predicciones)