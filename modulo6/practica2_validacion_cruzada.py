# Train/Test Split (Hold-out)
# K-Fold Cross-Validation
# Stratified K-Fold Cross-Validation
# Leave-One-Out (LOO) Cross-Validation

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    train_test_split, cross_val_score, accuracy_score,  precision_score, recall_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,accuracy_score, precision_score, recall_score

#1. Crear  datos simulado
X, y = make_classification(n_samples=300, n_features=5, n_informative=4, n_redundant=0, random_state=42,  flip_y=0.02, class_sep=1.0)
#X = array de tamana (300,5)
#y= un array de tamano (300) objetivo (0,1)
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

#2, Diccionar para almecenar resultado
metricas= ["Acurracy", "Precission", "Recall", "F1-score"]
resultado = {m: [] for m in metricas}
nombres = []

#3 . Metodo train/Test Split( Hold-out)
#Dividir datos en 80% para entrenamiento, 20% para prueba
X_train ,X_test ,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Crear y entrenar el modelo regresion lineal
model.fit(X_train,y_train)

#Predicir el conjuto de pruebas
y_pred = model.predict(X_test)



