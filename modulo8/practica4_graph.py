# =============================================================
# MNIST con TensorFlow — Visualizar dígitos mal clasificados
# =============================================================
# Este script:
# 1) Carga y prepara MNIST (28x28 → 784, normalizado).
# 2) Entrena una MLP sencilla (784→128→64→10) con Adam.
# 3) Predice en test y localiza los errores.
# 4) Muestra una rejilla (3x4) con imágenes mal clasificadas:
#    título "Real=X, Pred=Y", similar a tu ejemplo visual.
# 5) Guarda la figura como 'ejemplos_mal_clasificados.png'.

import numpy as np                    # Cálculo numérico básico
import matplotlib.pyplot as plt       # Gráficas con Matplotlib
import tensorflow as tf               # Framework de DL
from tensorflow.keras import layers, models  # Capas y modelos de Keras

# ---------- Semillas para reproducibilidad ----------
np.random.seed(42)                    # Fijamos semilla NumPy
tf.random.set_seed(42)                # Fijamos semilla TensorFlow

# ---------- 1) Cargar datos ----------
# Carga MNIST: (x_train, y_train) y (x_test, y_test)
# x_* tienen forma (num_imgs, 28, 28) con valores 0..255 (uint8)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Guardamos copia de imágenes de test SIN aplanar para graficarlas luego
x_test_imgs = x_test.copy()           # (10000, 28, 28)

# ---------- Preprocesamiento ----------
# Aplanamos imágenes a vectores de 784 y normalizamos a [0,1]
x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test  = x_test.reshape(-1, 28 * 28)  / 255.0

# ---------- 2) Definir el modelo ----------
# Modelo secuencial: capas apiladas en orden
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),  # Capa oculta 1
    layers.Dense(64, activation='relu'),                       # Capa oculta 2
    layers.Dense(10, activation='softmax')                     # Salida 10 clases
])

# ---------- 3) Compilar ----------
# Adam: optimizador adaptativo; pérdida para etiquetas enteras (0..9)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- 4) Entrenar ----------
# epochs=5 suele bastar para ~97% accuracy; validation_split=0.1 reserva 10% del train
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# ---------- 5) Evaluar ----------
# Calculamos pérdida y exactitud en el conjunto de prueba (no visto en entrenamiento)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy en test: {test_acc:.4f}")

# ---------- 6) Predicción ----------
# Probabilidades por clase (10 columnas). Batch grande acelera la inferencia.
probas = model.predict(x_test, batch_size=512, verbose=0)  # (10000, 10)
y_pred = np.argmax(probas, axis=1)                          # Clase predicha (entero 0..9)

# ---------- 7) Encontrar mal clasificados ----------
mis_idx = np.where(y_pred != y_test)[0]     # Índices donde la predicción es distinta a la etiqueta real
num_to_show = min(12, len(mis_idx))         # Mostraremos hasta 12 ejemplos (3x4)
print(f"Total mal clasificados: {len(mis_idx)} | Mostrando: {num_to_show}")

# En caso (raro) de cero errores, salimos limpiamente
if num_to_show == 0:
    print("¡No hubo ejemplos mal clasificados! (vuelve a entrenar con menos épocas o cambia la red)")
    raise SystemExit

# ---------- 8) Dibujar rejilla 3x4 ----------
rows, cols = 3, 4                           # Tamaño de la rejilla
fig, axes = plt.subplots(rows, cols, figsize=(12, 9))  # Figura y ejes
fig.suptitle("Ejemplos de dígitos mal clasificados", fontsize=16)  # Título general

# Recorremos los primeros 'num_to_show' índices con error
for i in range(num_to_show):
    idx = mis_idx[i]                         # Índice absoluto en el set de test
    ax = axes[i // cols, i % cols]           # Eje correspondiente en la rejilla (fila, col)
    ax.imshow(x_test_imgs[idx], cmap='viridis')  # Mostramos la imagen 28x28 (sin aplanar)
    ax.set_title(f"Real={y_test[idx]}, Pred={y_pred[idx]}")  # Título por panel
    ax.axis('off')                            # Ocultamos ejes para que se vea limpio

# Si no llenamos todos los subplots (p.ej. hay menos de 12 errores), apagamos el resto
for j in range(num_to_show, rows * cols):
    axes[j // cols, j % cols].axis('off')     # Desactiva ejes sobrantes

# Ajustamos espacios para que el título no choque con los subplots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Guardamos la figura a disco y la mostramos en pantalla
plt.savefig("ejemplos_mal_clasificados.png", dpi=150)  # Guarda PNG en el directorio actual
plt.show()                                             # Muestra la ventana gráfica
print("Figura guardada en 'ejemplos_mal_clasificados.png'")
