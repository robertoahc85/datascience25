# =============================================================
# CLASIFICACIÓN DE DÍGITOS MANUSCRITOS (MNIST) CON TENSORFLOW
# + Visualizaciones: curvas, matriz de confusión y ejemplos
# =============================================================

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# 1. Cargar datos MNIST
# -------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Guarda una copia 28x28 para visualizaciones posteriores
x_test_imgs = x_test.copy()

# Normalizar y aplanar imágenes
x_train = x_train.reshape(-1, 28*28) / 255.0
x_test  = x_test.reshape(-1, 28*28)  / 255.0

# -------------------------------
# 2. Definir la red neuronal
# -------------------------------
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(784,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# -------------------------------
# 3. Compilar el modelo
# -------------------------------
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# 4. Entrenar el modelo
# -------------------------------
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -------------------------------
# 5. Evaluar el modelo
# -------------------------------
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Accuracy en test: {acc:.4f} | Loss en test: {loss:.4f}")

# -------------------------------
# 6. Predicciones y tabla de resultados (20 primeras)
# -------------------------------
y_pred_probs = model.predict(x_test[:20], verbose=0)
y_pred_classes_20 = np.argmax(y_pred_probs, axis=1)

tabla = pd.DataFrame({
    "Índice": np.arange(20),
    "Etiqueta Real": y_test[:20],
    "Predicción": y_pred_classes_20
})
print("\nTabla de resultados (primeras 20 predicciones):")
print(tabla.to_string(index=False))

# ============================================================
# 7. VISUALIZACIONES
# ============================================================

# 7.1 Curvas de entrenamiento: pérdida
plt.figure(figsize=(6,4))
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Época')
plt.ylabel('Pérdida (loss)')
plt.title('Curva de Pérdida (train vs val)')
plt.legend()
plt.tight_layout()
plt.show()

# 7.2 Curvas de entrenamiento: accuracy
plt.figure(figsize=(6,4))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel('Época')
plt.ylabel('Exactitud (accuracy)')
plt.title('Curva de Accuracy (train vs val)')
plt.legend()
plt.tight_layout()
plt.show()

# 7.3 Matriz de confusión (sobre todo el test)
y_pred_probs_all = model.predict(x_test, verbose=0)
y_pred_classes_all = np.argmax(y_pred_probs_all, axis=1)

cm = confusion_matrix(y_test, y_pred_classes_all)
print("\nReporte de clasificación (test):")
print(classification_report(y_test, y_pred_classes_all, digits=4))

plt.figure(figsize=(6,5))
plt.imshow(cm, interpolation='nearest')
plt.title('Matriz de Confusión (Test)')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)

# Anotar valores dentro de la matriz
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.show()

# 7.4 Panel de ejemplos: 16 imágenes con predicción (aciertos y errores mezclados)
# Selecciona 16 índices (primeros 8 aciertos + 8 errores para balance visual)
correct_idx = np.where(y_pred_classes_all == y_test)[0][:8]
wrong_idx   = np.where(y_pred_classes_all != y_test)[0][:8]
panel_idx   = np.concatenate([correct_idx, wrong_idx])

plt.figure(figsize=(10,8))
for k, idx in enumerate(panel_idx):
    plt.subplot(4, 4, k+1)
    plt.imshow(x_test_imgs[idx], cmap='gray')
    pred = y_pred_classes_all[idx]
    true = y_test[idx]
    prob = y_pred_probs_all[idx, pred]
    title = f"Pred: {pred} ({prob:.2f})\nReal: {true}"
    plt.title(title, fontsize=9)
    plt.axis('off')
plt.suptitle('Muestras de Test — Predicción (prob) vs Real', y=0.96)
plt.tight_layout()
plt.show()
