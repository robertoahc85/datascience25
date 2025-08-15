# Visualizacion de datos de balanceo:
#
#

#
import os
import collections
from collections import Counter  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split  

#Tecnicas de balanceo:
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.combine import SMOTEENN
from sklearn.datasets import make_classification

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
    ax[0].bar(list(cnt_b.keys()), list(cnt_b.values()))
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
    fig = plt.figure(figsize=(10, 4))
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
    print("Distribucion de clase original", cnt_original )
    
# 2 ) Split estratificado ( para simular entrenamienti real) 
    X_train, X_test, y_train , y_test = train_test_split(X,y ,test_size= TEST_SIZE, stratify= y, random_state = RANDOM_STATE)
    
    print ("Muestra del Train antes del balanceo")
    print(X_train.head())
    cnt_train = Counter(y_train)
    print("Distribucion  TRAIN antes del balanceo", cnt_train)

#3 Balanceo sobre el train (nunca sobre Test)
# a) Under Sampling
    rus = RandomUnderSampler(sampling_strategy = UNDERSAMPLING_STRATEGY , random_state= RANDOM_STATE)
    X_under , y_under = rus.fit_resample(X_train, y_train)
    print("Muestra train Undersampling")
    print(X_under.head())
    cnt_under = Counter(y_under)
    print("Distribucion TRAIN despues del undersampling",cnt_under)
 #  b) Oversampling
    rus = RandomOverSampler(sampling_strategy = OVERSAMPLING_STRATEGY , random_state= RANDOM_STATE)
    X_over , y_over = rus.fit_resample(X_train, y_train)
    print("Muestra train Undersampling")
    print(X_over.head())
    cnt_over = Counter(y_over)
    print("Distribucion TRAIN despues del UnderSampling",cnt_over)
#  b) SMOTE
    rus = SMOTE(sampling_strategy = SMOTE_STRATEGY , k_neighbors = SMOTE_K_NEIGHBORS ,random_state= RANDOM_STATE)
    X_sm , y_sm = rus.fit_resample(X_train, y_train)
    print("Muestra train SMOTE")
    print(X_sm.head())
    cnt_sm = Counter(y_sm)
    print("Distribucion TRAIN despues del SMOTE",cnt_sm)   
    
    
    #4) grafica comparativa (antes , despues por tecnica)
    plot_side_by_side_counts("UnderSampling(antes vs despues)",y_train,y_under)   
    plot_side_by_side_counts("OverSampling(antes vs despues)",y_train,y_over)  
    plot_side_by_side_counts("SMOTES(antes vs despues)",y_train,y_sm)   
    
    # 5) grafica resumen ()
    resumen = {
        "Original (total)": {"class0": cnt_original.get(0,0), "class1": cnt_original.get(1,0)},
        "Train": {"class0": cnt_train.get(0, 0), "class1": cnt_train.get(1, 0)},
        "UnderSampling": {"class0": cnt_under.get(0, 0), "class1": cnt_under.get(1, 0)},
        "OverSampling": {"class0": cnt_over.get(0, 0), "class1": cnt_over.get(1, 0)},
        "SMOTE": {"class0": cnt_sm.get(0, 0), "class1": cnt_sm.get(1, 0)},
    }
    plot_sumary_counts(resumen)

if __name__ == '__main__':
    main()    