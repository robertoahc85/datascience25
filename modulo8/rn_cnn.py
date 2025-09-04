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
# Redes Neuronales Convolucionales
#
# Las redes neuronales convolucionales son una extensión de las redes neuronales,
# consideradas métodos de aprendizaje profundo o Deep Learning. Son generalmente usadas
# en procesamiento de imágenes para tareas como:
# - Análisis de imágenes
# - Clasificación de imágenes
# - Lenguaje Natural
#
# La arquitectura de una red convolucional se basa en una secuencia de capas de neuronas,
# alternando entre capas convolucionales y capas de agrupación.

# Convolución y capas de convolución
#
# La convolución es una técnica utilizada en procesamiento de señales e imágenes. En
# imágenes, se aplican filtros predefinidos para resaltar o disminuir ciertas características.
# Los filtros se aplican en un barrido por toda la imagen para producir una imagen alterada.

# Filtros
#
# En las redes de convolución, los filtros no están definidos manualmente. Los filtros son
# "aprendidos" durante el entrenamiento, resaltando las características más importantes de
# la imagen. Los filtros pueden capturar conceptos abstractos como bordes, siluetas, o
# características más complejas como ojos o bocas con capas adicionales.
#
# Parámetros importantes:
# - filters: Número de filtros generados en la capa.
# - padding: Forma de la imagen resultante tras el barrido.
# - kernel_size: Tamaño de los filtros creados.
# - strides: Número de píxeles ignorados entre movimientos del barrido.
#
# Documentación: https://keras.io/layers/convolutional/

# Capa de agrupación
#
# La capa de agrupación recorre la imagen aplicando una función (promedio o máximo) a cada
# sección para agrupar las características más relevantes.
#
# Documentación: https://keras.io/layers/pooling/

# =============================================================
# SECCIÓN DE CÓDIGO PYTHON
# -------------------------------------------------------------
# Load and display sample image for convolution demonstration
try:
    image = imread("data/arbol.png")
except FileNotFoundError:
    raise FileNotFoundError("Could not find 'data/arbol.png'. Ensure the file exists.")
print(f"Image shape: {image.shape}")
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis('off')
plt.show()

# Convert to grayscale
def to_greyscale(a):
    """
    Convert a pixel to grayscale using the luminance formula.
    
    Parameters:
        a (np.ndarray): Array of shape (3,) for RGB or (4,) for RGBA.
    
    Returns:
        float: Grayscale value.
    
    Raises:
        ValueError: If input has an unsupported number of channels.
    """
    if a.shape[0] == 3:  # RGB
        return 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]
    elif a.shape[0] == 4:  # RGBA
        return 0.299 * a[0] + 0.587 * a[1] + 0.114 * a[2]  # Ignore alpha channel
    elif a.shape[0] == 1:  # Grayscale
        return a[0]
    else:
        raise ValueError(f"Unsupported number of channels: {a.shape[0]}. Expected 1, 3, or 4.")

# Apply grayscale conversion
if len(image.shape) == 3:  # Check if image is not already grayscale
    bw_image = np.apply_along_axis(to_greyscale, 2, image)
else:
    bw_image = image  # Assume already grayscale
plt.figure(figsize=(6, 6))
plt.imshow(bw_image, cmap=plt.cm.gray)
plt.axis('off')
print(f"Grayscale image shape: {bw_image.shape}")
plt.show()

# Apply smoothing filter
smooth_filter = np.full((3, 3), 1/9)  # 3x3 averaging filter
new_image = signal.convolve2d(bw_image, smooth_filter, mode='same')
plt.figure(figsize=(6, 6))
plt.imshow(new_image, cmap=plt.cm.gray)
plt.axis('off')
print(f"Smoothed image shape: {new_image.shape}")
plt.show()

# Apply sharpening filter
kernel_filter = np.array([[-1/9, -1/9, -1/9],
                          [-1/9, 17/9, -1/9],
                          [-1/9, -1/9, -1/9]])
new_image = signal.convolve2d(bw_image, kernel_filter, mode='same')
plt.figure(figsize=(6, 6))
plt.imshow(new_image, cmap=plt.cm.gray)
plt.axis('off')
print(f"Sharpened image shape: {new_image.shape}")
plt.show()

# Apply Sobel filters for edge detection
kernel_filter_horizontal = np.array([[-1, -2, -1],
                                    [ 0,  0,  0],
                                    [ 1,  2,  1]])
new_image1 = signal.convolve2d(bw_image, kernel_filter_horizontal, mode='same')

kernel_filter_vertical = np.array([[-1,  0,  1],
                                  [-2,  0,  2],
                                  [-1,  0,  1]])
new_image2 = signal.convolve2d(bw_image, kernel_filter_vertical, mode='same')

# Display edge detection results
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.axis('off')
plt.title("Horizontal Edge Detection")
plt.imshow(new_image1, cmap=plt.cm.gray)
plt.subplot(122)
plt.axis('off')
plt.title("Vertical Edge Detection")
plt.imshow(new_image2, cmap=plt.cm.gray)
plt.tight_layout()
plt.show()
