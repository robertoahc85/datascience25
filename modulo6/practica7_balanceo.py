#Problema: el problema  desbalanceo de datos en la deteccio de fraude con tarjetas de credito
#Fradulente(Clase minoritaria) son muy escasas  en compracion con las transacciones
#legitimas(Clase mayoritaria). Este desbalanceo  provoca que lo modelos de clasificacion tienda 
# predecir mayoritariamente( Clase mayoritaria) y no detecte las transacciones fraudulentas.
#
#El objetivo es evaluar diferentes tecnicas de balanceo de datos para mejorar la deteccion de fraude.
#Se evaluaran las siguientes tecnicas de balanceo:
#1. Submuestreo aleatorio de la clase mayoritaria 
#2. Sobremuestreo aleatorio de la clase minoritaria 
# 3. Sobremuestreo con SMOTE 
# 4. Sobremuestreo con SmoTeen
#
#Se entrenara los modelo con cada version de los datos balanceados y se evaluara su rendimiento y usando metricas
#recall,f1-score y accuracy, matriz de confusion y curva ROC-AUC.
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