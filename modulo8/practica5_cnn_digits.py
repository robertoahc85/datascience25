# =============================================================
# SECCIÓN DE CÓDIGO PYTHON
# -------------------------------------------------------------
# Import necessary libraries
import matplotlib.pyplot as plt
from matplotlib.image import imread  # Use matplotlib.image.imread for compatibility
import numpy as np
from scipy import signal
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import seaborn as sns
from sklearn.metrics import confusion_matrix
# =============================================================
# SECCIÓN TEÓRICA / EDUCATIVA
# -------------------------------------------------------------
# Lectura de los datos
#
# ¿Cuáles son las características importantes por resaltar para la clasificación de imágenes?

# =============================================================
# SECCIÓN DE CÓDIGO PYTHON
# -------------------------------------------------------------
# Load and preprocess MNIST dataset
try:
    df = pd.read_csv('data/digit/train.csv')
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'data/digit/train.csv'. Ensure the file exists.")
print(f"El conjunto de datos tiene {df.shape[0]} filas y {df.shape[1]} columnas")
print(df.head())

# Extract features and labels
y = df['label'].values
x = df.drop("label", axis=1).values / 255.0  # Normalize pixel values
x = x.reshape((-1, 28, 28, 1))  # Reshape for CNN input

# Visualize 25 sample digits
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    number = x[i].reshape(28, 28)
    plt.imshow(number, cmap=plt.cm.binary)
    plt.xlabel(y[i])
plt.tight_layout()
plt.show()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=103)
print(f"x_train shape: {x_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# =============================================================
# SECCIÓN TEÓRICA / EDUCATIVA
# -------------------------------------------------------------
# Construyendo y entrenando la NN

# =============================================================
# SECCIÓN DE CÓDIGO PYTHON
# -------------------------------------------------------------
# Define CNN model
model = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding="same", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu', padding="same"),
    layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(rate=0.4),
    layers.Dense(32, activation='relu'),
    layers.Dropout(rate=0.4),
    layers.Dense(10, activation='softmax')
])
model.summary()

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=100, callbacks=[early_stopping])

# Evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

# =============================================================
# SECCIÓN TEÓRICA / EDUCATIVA
# -------------------------------------------------------------
# Probando la NN

# =============================================================
# SECCIÓN DE CÓDIGO PYTHON
# -------------------------------------------------------------
# Predict and analyze errors
predictions = model.predict(x_test)
y_hat = np.argmax(predictions, axis=1)
errores = x_test[y_test != y_hat]
errores_count = errores.shape[0]
print(f"Elementos de prueba: {y_test.shape[0]}")
print(f"Errores identificados: {errores_count}")
print(f"Porcentaje de error: {errores_count * 100 / y_test.shape[0]:.2f}%")

real_labels = y_test[y_test != y_hat]
predicted_labels = y_hat[y_test != y_hat]

# Visualize misclassified samples
if errores_count == 0:
    print("No misclassifications found.")
else:
    for j in range((errores_count + 4) // 5):  # Ceiling division
        plt.figure(figsize=(15, 3))
        for i in range(min(5, errores_count - 5*j)):
            plt.subplot(1, 5, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            number = errores[5*j+i].reshape(28, 28)
            plt.imshow(number, cmap=plt.cm.binary)
            plt.xlabel(f"Real: {real_labels[5*j+i]} Pred: {predicted_labels[5*j+i]}")
        plt.tight_layout()
        plt.show()

# =============================================================
# SECCIÓN TEÓRICA / EDUCATIVA
# -------------------------------------------------------------
# Matriz de confusión

# =============================================================
# SECCIÓN DE CÓDIGO PYTHON
# -------------------------------------------------------------
# Compute and visualize confusion matrix
mod_confusion_matrix = confusion_matrix(y_test, y_hat)
for i in range(10):
    mod_confusion_matrix[i][i] = 0  # Set diagonal to zero to highlight misclassifications

plt.figure(figsize=(12, 10))
sns.heatmap(mod_confusion_matrix, linewidth=0.5, annot=True, fmt='d', cmap="YlGnBu")
plt.ylabel("Etiquetas Reales")
plt.xlabel("Etiquetas Predecidas")
plt.show()

# =============================================================