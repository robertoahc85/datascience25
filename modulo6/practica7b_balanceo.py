# Visualizacion de datos de balanceo:
#
#

#
import os
import collections
from typing import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  sklearn.model_selection as train_test_split
import sklearn.datasets as make_classification

#Tecnicas de balanceo:
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN

# ---------------------------------------------
#Parametros  Ajustables
# ---------------------------------------------
DATA_PATH  = 'creditcard.csv'  # Ruta al archivo CSV descargado
TEST_SIZE = 0.2  # Proporción del conjunto de prueba
RANDOM_STATE = 42  # Semilla para reproducibilidad

# ---------------------------------------------
#Tamano/  de balanceo   
# ---------------------------------------------
UNDERSAMPLING_STRATEGY = 1.0     # 1.0  -> IGUAL A la clase  A LA MAYORÍA
OVERSAMPLING_STRATEGY = 0.30    # 0.30   -> minoritaria de 30% de la mayoría
SMOTE_STRATEGY =0.30           # 0.30 -> minoritaria de 30% de la mayoría
SMOTE_K_NEIGHBORS = 3  # Número de vecinos para SMOTE

# ---------------------------------------------
# Cargar los datos reales y sintentico
# ---------------------------------------------
def load_data(data_path, random_state=RANDOM_STATE):
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
        print(f"Datos cargados desde {data_path}")
        if 'Class' not in data.columns:
            raise ValueError("El archivo real debe contener la columna Class(1=Fraude, 0= legitima).")
        X= data.drop(columns=['Class'])
        y = data['Class'].astype(int)  # Asegurarse de que 'Class' sea de tipo int
        HAVE_REAL = True
    else:
        X_np, y_np = make_classification(
        n_classes=2,
        class_sep=2,
        weights=[0.9, 0.1],
        n_redundant=0,
        n_features=12,
        n_clusters_per_class=1,
        n_samples=5000,
        random_state=random_state)
        X = pd.DataFrame(X_np, columns=[f'feature_{i}' for i in range(X_np.shape[1])])
        y = pd.Series(y_np, name='Class')
        HAVE_REAL = False
    return X, y, HAVE_REAL


# ---------------------------------------------
# utilidades de visualizacion
# ---------------------------------------------
def plot_side_by_side_counts(title, y_before, y_after):
    """
    Plots side by side counts of two series.
    """
    cnt_b = Counter(y_before)
    cnt_a = Counter(y_after) 
    
    fig, ax = plt. subplots(1,2 , figsize=(8,4))
    
    #Antes
    ax[0].bar(list(cnt_b.keys(),list(cnt_b.values())))
    ax[0].set_title("Antes")
    ax[0].set_xticks([0,1])
    ax[0].set_xlabel("Clase")
    ax[0].set_ylabel("Cantidad")

    # Después
    ax[1].bar(list(cnt_a.keys()), list(cnt_a.values()))
    ax[1].set_title("Después")
    ax[1].set_xticks([0,1])
    ax[1].set_xlabel("Clase")
    ax[1].set_ylabel("Cantidad")

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def plot_sumary_counts(sumary_dict):
    #Graficpo resumen con la cantidad de clase 1 (fraudes) para:
    #Original  , Train, UnderSlamping , Oversampling, SMOTE.
    
    label = list(sumary_dict.keys()) 
    vals_0 = [sumary_dict[k]['class0'] for k in label]
    vals_1 = [sumary_dict[k]['class1'] for k in label]
    
    #una grafica por clase
    fig = plt.figure(figsize=(10.4))
    #Clase 0
    ax1= fig.add_subplot(121)
    ax1.bar(label, vals_0)
    ax1.set_title("Clase 0 (Legítima)")
    ax1.set_ylabel("Cantidad")
    ax1.set_xticklabels(label, rotation=30)

    # Clase 1
    ax2 = fig.add_subplot(122)
    ax2.bar(label, vals_1, color='orange')
    ax2.set_title("Clase 1 (Fraude)")
    ax2.set_ylabel("Cantidad")
    ax2.set_xticklabels(label, rotation=30)

    fig.suptitle("Resumen de cantidad de muestras por clase")
    plt.tight_layout()
    plt.show()
    


def main():
#1) cargoa los datos
    X ,y, HAVE_REAL= load_data(DATA_PATH)
    
    print ("Muestra de datos reales")
    if HAVE_REAL:
        print(pd.concat([X,y], axis=1).head())
    else :
        print(pd.concat([X,y], axis=1).head())    
        
    cnt_original= Counter(y)   
    print("Distribucion de clase original", cnt_original   )
    
if __name__ == '__main__':
    main()    