# ============================================================
# DIGIT RECOGNIZER (Kaggle MNIST) — MLP desde cero
# ============================================================
# TEORÍA (resumen):
# - Objetivo: clasificar imágenes de dígitos escritos a mano (0..9).
# - Datos: cada imagen es 28x28 (=784) pixeles en escala de grises (0..255).
# - Enfoque: Red neuronal de 2 capas ocultas con activación sigmoide.
# - Por qué ONE-HOT: convierte la etiqueta entera (p.ej. 7) a un vector
#   categórico sin orden [0,0,0,0,0,0,0,1,0,0]; evita inducir ordinalidad.
# - Por qué normalizar: escalar pixeles a [0,1] acelera y estabiliza el
#   entrenamiento (las activaciones no saturan tan fácil).
# - Backprop: regla de la cadena para propagar el gradiente salida→ocultas→entrada.
# - Nota: para multiclase lo más estándar es salida softmax + cross-entropy;
#   aquí usamos sigmoide+MSE para seguir tu notebook original.
# ============================================================

# -----------------------------
# 1) IMPORTS
# -----------------------------
import numpy as np                     # Álgebra lineal y utilidades numéricas
import pandas as pd                    # Manejo de tablas y CSV
import matplotlib.pyplot as plt        # Gráficas

# -----------------------------
# 2) CARGA DE DATOS
# -----------------------------
df = pd.read_csv('data/train.csv')     # Carga el CSV (ajusta la ruta si hace falta)

y = df.label.values                    # y: vector de enteros 0..9 (N,)
x = df.drop("label", axis=1)           # X crudo: DataFrame con 784 columnas de pixeles

print("El conjunto de datos tiene {} filas y {} columnas".format(x.shape[0], x.shape[1]))
print(x.head(2))                       # Vistazo rápido a las primeras filas

# -----------------------------
# 3) ONE-HOT ENCODING
# -----------------------------
def one_hot(j: int) -> np.ndarray:
    """
    Convierte un entero j en un vector one-hot de tamaño 10.
    TEORÍA: evita que el modelo 'crea' que 9 > 3; todas las clases
    se vuelven categóricas y mutuamente excluyentes.
    """
    e = np.zeros(10, dtype=np.float32) # vector [0,0,...,0] de longitud 10
    e[j] = 1.0                         # coloca el 1 en la posición de la clase
    return e

y_onehot = np.array([one_hot(n) for n in y], dtype=np.float32)  # (N,10)

# -----------------------------
# 4) NORMALIZACIÓN DE FEATURES
# -----------------------------
X = x.values.astype(np.float32) / 255.0 # escala pixeles 0..255 → 0..1 (mejor para sigmoide)
Y = y_onehot                            # etiquetas en formato one-hot

# -----------------------------
# 5) ACTIVACIONES (SIGMOID)
# -----------------------------
def sigmoid(t):
    """σ(t) = 1 / (1 + e^-t). TEORÍA: acota la salida en (0,1)."""
    return 1.0 / (1.0 + np.exp(-t))

def sigmoid_derivative(p):
    """
    Derivada de sigmoide cuando p = σ(t): σ'(t) = σ(t)*(1-σ(t))
    TEORÍA: necesaria para backprop. Si pasas preactivación, usa σ(z) primero.
    """
    return p * (1.0 - p)

# -----------------------------
# 6) INICIALIZACIÓN DE PESOS
# -----------------------------
lr = 0.01                               # tasa de aprendizaje (hiperparámetro)
# Nota teórica: pesos pequeños ayudan a no saturar sigmoides al inicio.
W0 = np.random.randn(X.shape[1], 16).astype(np.float32)  # (784,16) capa1
W1 = np.random.randn(16, 16).astype(np.float32)          # (16,16)  capa2
W2 = np.random.randn(16, 10).astype(np.float32)          # (16,10)  salida

b0 = np.random.randn(1, 16).astype(np.float32)           # sesgos capa1
b1 = np.random.randn(1, 16).astype(np.float32)           # sesgos capa2
b2 = np.random.randn(1, 10).astype(np.float32)           # sesgos salida

mse = 0.0                               # para registrar pérdida MSE
errors = []                             # historial de pérdidas

# -----------------------------
# 7) FEEDFORWARD
# -----------------------------
def feedforward(x_input):
    """
    Propagación hacia adelante: calcula salida de la red para x_input.
    Guarda intermedios globales para backprop.
    """
    global a0, z0, a1, z1, a2, z2, a3, output
    global W0, W1, W2, b0, b1, b2

    a0 = x_input                        # a0: activaciones de entrada (B,784)
    z0 = np.dot(a0, W0) + b0            # preactivación capa1 (B,16)
    a1 = sigmoid(z0)                    # activación capa1 (B,16)

    z1 = np.dot(a1, W1) + b1            # preactivación capa2 (B,16)
    a2 = sigmoid(z1)                    # activación capa2 (B,16)

    z2 = np.dot(a2, W2) + b2            # preactivación salida (B,10)
    a3 = sigmoid(z2)                    # activación salida (B,10) — probs 'tipo' sigmoide
    output = a3                         # alias claro

    return output                       # retorna ŷ (B,10)

# -----------------------------
# 8) BACKPROPAGATION
# -----------------------------
def backprop(y_n):
    """
    Retropropaga el error y actualiza pesos/sesgos.
    TEORÍA: regla de la cadena — gradiente de la pérdida fluye de la salida
    hacia capas anteriores multiplicando derivadas locales.
    """
    global mse, W0, W1, W2, b0, b1, b2
    global a0, a1, a2, z0, z1, z2, output
    global lr, errors

    # Pérdida MSE por batch (como en tu notebook); para CE usa otra fórmula.
    mse = np.sum((y_n - output) ** 2)   # suma (no media) para seguir tu celda
    errors.append(mse)                  # guarda historial

    # Gradiente en la salida: dL/dz2 = -(y - ŷ) * σ'(z2)
    # Nota: usamos la versión que toma la activación ya aplicada (output).
    delta2 = -(y_n - output) * sigmoid_derivative(output)

    # Gradientes de pesos/sesgo de salida
    d_w2 = np.dot(a2.T, delta2)                         # (16,10)
    d_b2 = delta2                                       # (B,10)

    # Propaga a capa oculta 2: delta1 = (delta2 @ W2^T) * σ'(z1)
    delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(a2)
    d_w1 = np.dot(a1.T, delta1)                         # (16,16)
    d_b1 = delta1                                       # (B,16)

    # Propaga a capa oculta 1: delta0 = (delta1 @ W1^T) * σ'(z0)
    delta0 = np.dot(delta1, W1.T) * sigmoid_derivative(a1)
    d_w0 = np.dot(a0.T, delta0)                         # (784,16)
    d_b0 = delta0                                       # (B,16)

    # Actualización de pesos (descenso por gradiente)
    W2 -= lr * d_w2;  W1 -= lr * d_w1;  W0 -= lr * d_w0

    # Actualización de sesgos usando promedio por batch (más estable)
    b2 -= lr * d_b2.mean(axis=0, keepdims=True).reshape(b2.shape)
    b1 -= lr * d_b1.mean(axis=0, keepdims=True).reshape(b1.shape)
    b0 -= lr * d_b0.mean(axis=0, keepdims=True).reshape(b0.shape)

# -----------------------------
# 9) VISUALIZACIÓN INICIAL
# -----------------------------
i = 0                                    # índice de la primera imagen
img = X[i].reshape(28, 28)               # reordena 784→28x28 para mostrar

plt.figure(figsize=(7, 7))               # figura grande
plt.imshow(img, cmap="viridis")          # muestra la imagen
plt.title(f"El número escrito es: {y[i]}")  # título con la etiqueta real
# Escribimos TODOS los valores de píxel (incluyendo ceros):
for r in range(28):
    for c in range(28):
        val_0_255 = int(img[r, c] * 255) # reconvertimos a 0..255 para leer mejor
        plt.text(c, r, str(val_0_255), ha="center", va="center", color="white", fontsize=5)
plt.axis("off")                          # oculta ejes
plt.show()                               # renderiza

# Cuadrícula con los primeros 10 dígitos (solo color, sin valores sobreimpresos)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
for j, ax in enumerate(axes.flat):
    ax.imshow(X[j].reshape(28, 28), cmap="viridis")
    ax.set_title(f"{y[j]}", fontsize=12) # título: etiqueta real
    ax.axis("off")
plt.suptitle("Primeros 10 dígitos del dataset", fontsize=14)
plt.show()

# -----------------------------
# 10) "ENTRENAMIENTO" RÁPIDO (una pasada)*
# -----------------------------
# *Para replicar tus celdas y tener algo que evaluar, ejecutamos una sola
#   iteración sobre todo X. Para un entrenamiento real, recorre epochs y mini-batches.
_ = feedforward(X)   # forward de todo el dataset
backprop(Y)          # un paso de backprop con todas las muestras (full-batch)

# -----------------------------
# 11) PREDICCIÓN Y ERRORES
# -----------------------------
y_hat = feedforward(X).argmax(axis=1)     # clase predicha = índice con mayor activación
mask_err = (y != y_hat)                   # máscara booleana de errores
idx_err = np.where(mask_err)[0]           # índices de muestras mal clasificadas
print("Errores totales en el dataset:", len(idx_err))

# -----------------------------
# 12) VISUALIZACIÓN DE ERRORES EN CUADRÍCULA
# -----------------------------
if len(idx_err) > 0:
    k = min(12, len(idx_err))             # cuántos errores mostrar (hasta 12)
    sel = idx_err[:k]                     # selecciona los primeros k índices

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # cuadrícula 3x4
    axes = axes.flatten()                              # aplana para iterar fácil

    for ax, idx in zip(axes, sel):
        ax.imshow(x.iloc[idx].values.reshape(28, 28), cmap="viridis")  # imagen del error
        ax.set_title(f"Real={y[idx]}, Pred={y_hat[idx]}")              # etiqueta vs predicción
        ax.axis("off")                                                 # sin ejes

    plt.suptitle("Ejemplos de dígitos mal clasificados", fontsize=16)
    plt.tight_layout()
    plt.show()
