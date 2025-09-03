# ============================================================
# CNN con Keras para clasificar imágenes de CIFAR-10
# Incluye:
# - Preprocesamiento de datos
# - Arquitectura de red convolutiva simple
# - Aumento de datos con ImageDataGenerator
# - Entrenamiento y evaluación
# - Curvas de entrenamiento y matriz de confusión
# ============================================================

# 1) Importación de librerías necesarias
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# 2) Cargar y preprocesar el dataset CIFAR-10
# ------------------------------------------------------------
# CIFAR-10: 60,000 imágenes 32x32x3 en 10 clases (50k train / 10k test)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalizar píxeles al rango [0,1] para acelerar y estabilizar el entrenamiento
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convertir etiquetas a one-hot encoding (necesario para categorical_crossentropy)
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# ------------------------------------------------------------
# 3) Definir la arquitectura de la CNN
# ------------------------------------------------------------
model = keras.Sequential([
    # Primera capa convolucional + MaxPooling
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Segunda capa convolucional + MaxPooling
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Aplanado y Fully Connected
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 clases de salida
])

# Mostrar el resumen de la arquitectura
model.summary()

# ------------------------------------------------------------
# 4) Compilar el modelo
# ------------------------------------------------------------
# Se puede usar SGD o Adam. Aquí usamos Adam (más robusto).
model.compile(
    optimizer='adam',                        # también se puede usar 'SGD'
    loss='categorical_crossentropy',         # pérdida para clasificación multiclase
    metrics=['accuracy']                     # métrica principal: exactitud
)

# ------------------------------------------------------------
# 5) Aumento de datos (ImageDataGenerator)
# ------------------------------------------------------------
# Genera variaciones aleatorias para robustez y evita sobreajuste
datagen = ImageDataGenerator(
    rotation_range=20,       # rotaciones aleatorias
    width_shift_range=0.2,   # desplazamiento horizontal
    height_shift_range=0.2,  # desplazamiento vertical
    horizontal_flip=True     # voltear imágenes horizontalmente
)

# Ajustar el generador a los datos de entrenamiento
datagen.fit(x_train)

# ------------------------------------------------------------
# 6) Entrenamiento de la CNN
# ------------------------------------------------------------
# Usamos fit con flujo de imágenes aumentadas
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=10,
    validation_data=(x_test, y_test)
)

# ------------------------------------------------------------
# 7) Evaluación del modelo en test
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nPrecisión en test: {test_acc:.4f}")

# ------------------------------------------------------------
# 8) Gráficas de entrenamiento
# ------------------------------------------------------------

# Pérdida
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.title("Curva de Pérdida")
plt.show()

# Precisión
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.title("Curva de Precisión")
plt.show()

# ------------------------------------------------------------
# 9) Matriz de confusión
# ------------------------------------------------------------
# Predicciones sobre test
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcular matriz
cm = confusion_matrix(y_true, y_pred)

# Graficar matriz de confusión con seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title("Matriz de Confusión - CIFAR-10")
plt.show()
