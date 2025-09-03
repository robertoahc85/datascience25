#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================
# XOR con TensorFlow/Keras — MLP de 1 capa oculta
# ============================================
# Objetivo: aprender la función XOR sobre entradas binarias {0,1}×{0,1}.
# Teoría clave:
# - XOR NO es linealmente separable → se requiere al menos 1 capa oculta con
#   activación no lineal (ej. tanh, ReLU, sigmoid).
# - Usamos una red 2→4→1 (2 entradas, 4 neuronas ocultas, 1 salida).
# - Función de pérdida binaria y descenso de gradiente (SGD/Adam) con backprop.

# ---------- 1) Dependencias ----------
# pip install tensorflow matplotlib numpy
import numpy as np                                # Cálculo numérico y mallas para graficar
import matplotlib.pyplot as plt                   # Gráfica de la región de decisión
from tensorflow import keras                      # API de alto nivel (Keras) dentro de TensorFlow
from tensorflow.keras import layers               # Capas densas (fully connected), etc.
from tensorflow.keras.optimizers import SGD       # Optimizador por gradiente estocástico

# ---------- 2) Datos (XOR) ----------
# Arreglo de entradas: cuatro combinaciones posibles de dos bits.
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# Etiquetas binarias:
# XOR devuelve 1 cuando los bits son diferentes, y 0 cuando son iguales.
y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# ---------- 3) Definición del modelo ----------
# Modelo secuencial: capas apiladas linealmente.
# Capa oculta: 4 neuronas, activación 'tanh' (no lineal → permite modelar XOR).
#   - input_shape=(2,) indica que cada ejemplo tiene 2 características.
# Capa de salida: 1 neurona con 'sigmoid' (probabilidad ∈ (0,1) para clase 1).
model = keras.Sequential([
    layers.Dense(4, activation='tanh', input_shape=(2,)),  # 2→4
    layers.Dense(1, activation='sigmoid')                  # 4→1
])

# ---------- 4) Compilación ----------
# Optimizador: SGD (descenso de gradiente estocástico).
#   - learning_rate (antes 'lr') controla el tamaño del paso de actualización.
# Pérdida: binary_crossentropy (adecuada para clasificación binaria).
# Métricas: accuracy para monitorear desempeño.
sgd = SGD(learning_rate=0.1)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

# Imprimimos un resumen para ver capas, formas y conteo de parámetros.
print("\n=== Resumen del modelo ===")
model.summary()

# ---------- 5) Entrenamiento ----------
# epochs: cuántas pasadas completas por el dataset.
# batch_size=1: actualiza pesos tras cada ejemplo (aprendizaje más ruidoso pero funciona bien en XOR).
history = model.fit(X, y, epochs=500, batch_size=1, verbose=0)

# ---------- 6) Evaluación ----------
# Calcula pérdida y exactitud sobre los 4 puntos de entrenamiento (overfitting aquí no es problema).
loss, acc = model.evaluate(X, y, verbose=0)
print(f"\nPérdida (binary_crossentropy): {loss:.6f}")
print(f"Exactitud (accuracy): {acc:.4f}")

# ---------- 7) Predicciones sobre los 4 puntos ----------
# Debe aproximar: [0, 1, 1, 0]
y_hat = model.predict(X, verbose=0)
print("\nPredicciones sobre (0,0), (0,1), (1,0), (1,1):")
for inp, pred in zip(X, y_hat):
    print(f"X={inp} → ŷ={pred[0]:.4f} (umbral 0.5 → clase {int(pred[0] >= 0.5)})")

# ---------- 8) Cálculo de #neuronas y #pesos ----------
# Neuronas (excluyendo entradas): 4 ocultas + 1 de salida = 5 neuronas.
num_neuronas = 4 + 1

# Pesos/cant. parámetros entrenables:
#  - Capa 1 (2→4): 2*4 pesos + 4 sesgos = 12
#  - Capa 2 (4→1): 4*1 pesos + 1 sesgo = 5
#  Total = 17
params_l1 = (2 * 4) + 4
params_l2 = (4 * 1) + 1
total_params = params_l1 + params_l2

print(f"\nNeuronas (ocultas+salida): {num_neuronas}")
print(f"Parámetros capa 1 (2→4): {params_l1}")
print(f"Parámetros capa 2 (4→1): {params_l2}")
print(f"Parámetros totales: {total_params}")

# ---------- 9) Región de decisión (malla 2D) ----------
# Creamos una grilla de puntos en [0,1]×[0,1] para visualizar cómo el modelo separa clases.
x_0 = np.linspace(0, 1, 200)          # eje x (feature 1)
x_1 = np.linspace(0, 1, 200)          # eje y (feature 2)
X_0, X_1 = np.meshgrid(x_0, x_1)      # malla cartesiana
points = np.c_[X_0.ravel(), X_1.ravel()]  # (200*200, 2) → lista de puntos

# Predicción de probabilidad sobre toda la malla.
y_grid = model.predict(points, verbose=0).reshape(X_0.shape)

# ---------- 10) Gráfica ----------
fig, ax = plt.subplots(figsize=(6, 6))
# Contorno relleno de la probabilidad hacia la clase 1.
cntr = ax.contourf(X_0, X_1, y_grid, levels=50, cmap="plasma")
plt.colorbar(cntr, ax=ax, label="Probabilidad clase 1 (sigmoid)")

# Puntos de entrenamiento: azules para y=0, naranjas para y=1
ax.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], marker='o', edgecolor='k', label='Clase 0')
ax.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], marker='^', edgecolor='k', label='Clase 1')

ax.set_title("XOR — Región de decisión aprendida (MLP 2→4→1)")
ax.set_xlabel("x0")
ax.set_ylabel("x1")
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right')

# Guardamos la figura para verla fuera de Jupyter.
plt.tight_layout()
plt.savefig("xor_decision.png", dpi=150)
print("\nSe guardó la figura de la región de decisión en: xor_decision.png")
