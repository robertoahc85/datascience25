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
from sklearn.preprocessing import StandardScaler
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
def preparar_datos():
    data =cargar_datos()
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

###
#Modelo de Bosque Aleatorio (Random Forest)
#
#Proposito: Clasificacio o regresion robusta con múltiples árboles de decisión
#Como funciona: Genera múltiples árboles de decisión sobre subconjuntos aleatorios de datos y características,
#promediando sus predicciones para mejorar la precisión y reducir el sobreajuste.
def entrenar_bosque_aleatorio(X_train, X_test, y_train, y_test):
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    #Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    #Realizar predicciones de los datos de prueba
    predicciones = modelo.predict(X_test)
    #Graficar la matriz de confusión
    graficar_matriz_confusion(y_test, predicciones, 'Bosque Aleatorio')
    return modelo, predicciones, accuracy_score(y_test, predicciones)



#Modelo de Máquinas de Vectores de Soporte (SVM)
#Proposito: Clasificación o regresión minimizando el margen entre clases
#Como funciona: Encuentra el hiperplano óptimo que separa las clases con el mayor margen posible,
#utilizando un kernel para transformar los datos si es necesario.
def entrenar_svm(X_train, X_test, y_train, y_test):
    #Estandarizar las características
    escalador = StandardScaler()
    #Ajustar y transformar los datos de entrenamiento y transformar los datos de prueba
    X_train_escalado = escalador.fit_transform(X_train)
    #Transformar los datos de prueba con el mismo escalador
    X_test_escalado = escalador.transform(X_test)
    modelo = SVC(kernel='linear', random_state=42)
    #Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train_escalado, y_train)
    #Realizar predicciones de los datos de prueba
    predicciones = modelo.predict(X_test_escalado)
    #Graficar la matriz de confusión
    graficar_matriz_confusion(y_test, predicciones, 'Máquinas de Vectores de Soporte (SVM)')
    return modelo, predicciones, accuracy_score(y_test, predicciones)
  
def comparar_modelos(y_test, predicciones_dict,accuracies_dict):
    print("Comparación de Modelos:")
    for nombre_modelo, predicciones in predicciones_dict.items():
        print(f"\nModelo: {nombre_modelo}:{accuracies_dict[nombre_modelo]:.2f}")
        print(f"Reporte de Clasificación para{nombre_modelo}:")
        print(classification_report(y_test, predicciones))
        
    #Crear una grafica de barras para comparar las precisiones
    plt.figure(figsize=(10, 6))
    modelo = list(accuracies_dict.keys())
    precisiones = list(accuracies_dict.values())
    sns.barplot(x=modelo, y=precisiones)
    plt.ylim(0, 1)
    plt.ylabel('Precisión')
    plt.title('Comparación de Precisión entre Modelos')
    plt.show()      

def main():
    X, y = preparar_datos()
    #Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    #Entrenar y evaluar los modelos
    modelos = {
        'Regresión Logística': entrenar_regresion_logistica(X_train, X_test, y_train, y_test),
        'Árbol de Decisión': entrenar_arbol_decision(X_train, X_test, y_train, y_test),
        'Bosque Aleatorio': entrenar_bosque_aleatorio(X_train, X_test, y_train, y_test),
        'SVM': entrenar_svm(X_train, X_test, y_train, y_test)
    }

    #Comparar los modelos
    predicciones_dict = {nombre: modelo[1] for nombre, modelo in modelos.items()}
    accuracies_dict = {nombre: modelo[2] for nombre, modelo in modelos.items()}
    
    comparar_modelos(y_test, predicciones_dict, accuracies_dict)
    
if __name__ == "__main__":
        # Ejecutar la función principal
    main()