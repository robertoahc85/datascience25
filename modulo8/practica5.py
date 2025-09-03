# ============================================================
# Actividad Práctica Guiada (DOCUMENTADA):
# CNN para clasificar imágenes (CIFAR-10) con TensorFlow/Keras
# ------------------------------------------------------------
# Objetivo:
# - Entrenar una red neuronal convolutiva (CNN) para clasificar imágenes
#   del dataset CIFAR-10 en 10 clases.
#
# Qué aprenderás:
# - Preprocesamiento (normalización, split de validación)
# - Arquitectura CNN moderna (Conv2D+BN+ReLU+MaxPool+Dropout)
# - Aumento de datos (data augmentation)
# - Entrenamiento con callbacks (EarlyStopping / ReduceLROnPlateau)
# - Evaluación con métricas y visualizaciones
# - Ejemplos mal clasificados para análisis de error
#
# Dónde se aplican las CNN (al final del script se imprime una guía):
# - Visión por computadora: clasificación, detección, segmentación.
# - Medicina: análisis de rayos X, resonancias, retinografías.
# - Industria: inspección de calidad, conteo y seguimiento de objetos.
# - Autos autónomos: percepción (señales, peatones, carriles).
# - Satélites/agro: clasificación de cultivos, mapeo y monitoreo.
# - Seguridad: reconocimiento de personas/placas (cumpliendo normativa).
# - Retail: análisis de estanterías, reconocimiento de productos.
# ============================================================

# 1) Importar librerías
import numpy as np                            # Cálculo numérico
import matplotlib.pyplot as plt               # Gráficas
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf                       # DL framework
from tensorflow.keras import layers, models, callbacks

# ------------------- Reproducibilidad -----------------------
# Semillas fijas para tener resultados (razonablemente) repetibles
np.random.seed(42)
tf.random.set_seed(42)

# 2) Cargar y preprocesar el dataset
# CIFAR-10: 60k imágenes color (32x32x3) en 10 clases
# - 50k para entrenamiento, 10k para prueba
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.ravel()   # Keras entrega etiquetas con shape (N,1) → a vector plano
y_test  = y_test.ravel()

# Normalizar a [0,1] mejora la estabilidad numérica y la convergencia
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# Reservamos 10% del train como validación (para monitorear sobreajuste)
val_split = 0.1
num_val = int(len(x_train) * val_split)
x_val, y_val = x_train[:num_val], y_train[:num_val]
x_train, y_train = x_train[num_val:], y_train[num_val:]

# 3) Definir la arquitectura de la CNN
# Aumento de datos: genera ligeras variaciones para robustez (evita sobreajuste)
num_classes = 10
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),  # invierte horizontalmente (simetría común en objetos)
    layers.RandomRotation(0.05),      # rota ligeramente (robustez a orientación)
    layers.RandomZoom(0.1)            # zoom aleatorio (escala)
], name="augment")

def make_model():
    # Capa de entrada: tensor imagen 32x32 con 3 canales (RGB)
    inputs = layers.Input(shape=(32, 32, 3), name="input_image")

    # Aumento de datos SOLO en entrenamiento; Keras lo maneja internamente
    x = data_augmentation(inputs)

    # (Opcional) Rescaling identidad aquí por claridad: ya normalizamos antes,
    # la dejaremos como identidad para resaltar el "sitio" típico del reescalado.
    x = layers.Rescaling(1.0, name="rescale_identity")(x)

    # ------------------ Bloque Convolutivo 1 ------------------
    # Conv2D extrae filtros locales (bordes, texturas)
    x = layers.Conv2D(32, kernel_size=3, padding="same", name="b1_conv1")(x)
    x = layers.BatchNormalization(name="b1_bn1")(x)  # estabiliza distribución de activaciones
    x = layers.ReLU(name="b1_relu1")(x)

    x = layers.Conv2D(32, kernel_size=3, padding="same", name="b1_conv2")(x)
    x = layers.BatchNormalization(name="b1_bn2")(x)
    x = layers.ReLU(name="b1_relu2")(x)

    x = layers.MaxPooling2D(name="b1_pool")(x)  # reduce resolución espacial (invariancia local)
    x = layers.Dropout(0.25, name="b1_drop")(x) # regulariza, previene sobreajuste

    # ------------------ Bloque Convolutivo 2 ------------------
    x = layers.Conv2D(64, kernel_size=3, padding="same", name="b2_conv1")(x)
    x = layers.BatchNormalization(name="b2_bn1")(x)
    x = layers.ReLU(name="b2_relu1")(x)

    x = layers.Conv2D(64, kernel_size=3, padding="same", name="b2_conv2")(x)
    x = layers.BatchNormalization(name="b2_bn2")(x)
    x = layers.ReLU(name="b2_relu2")(x)

    x = layers.MaxPooling2D(name="b2_pool")(x)
    x = layers.Dropout(0.25, name="b2_drop")(x)

    # ------------------ Bloque Convolutivo 3 ------------------
    x = layers.Conv2D(128, kernel_size=3, padding="same", name="b3_conv1")(x)
    x = layers.BatchNormalization(name="b3_bn1")(x)
    x = layers.ReLU(name="b3_relu1")(x)

    x = layers.Conv2D(128, kernel_size=3, padding="same", name="b3_conv2")(x)
    x = layers.BatchNormalization(name="b3_bn2")(x)
    x = layers.ReLU(name="b3_relu2")(x)

    x = layers.MaxPooling2D(name="b3_pool")(x)
    x = layers.Dropout(0.30, name="b3_drop")(x)

    # -------------- Clasificador completamente conectado --------------
    x = layers.Flatten(name="flatten")(x)             # vectoriza mapas de características
    x = layers.Dense(256, name="fc1")(x)              # capa densa para combinar características
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.ReLU(name="fc1_relu")(x)
    x = layers.Dropout(0.40, name="fc1_drop")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    model = models.Model(inputs, outputs, name="cnn_cifar10")
    return model

# Construimos el modelo y vemos un resumen de capas/parametría
model = make_model()
model.summary()

# 4) Compilar y entrenar el modelo
# - Adam: optimizador adaptativo robusto
# - Pérdida: SCCE (SparseCategoricalCrossentropy) porque etiquetas son enteros 0..9
# - Métrica: accuracy (porcentaje de aciertos)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks:
# - EarlyStopping: si la val_accuracy no mejora por 'patience' épocas, detiene y restaura mejores pesos
early = callbacks.EarlyStopping(
    monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True
)
# - ReduceLROnPlateau: si la val_loss se estanca, reduce LR a la mitad (hasta min_lr)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5
)

# Entrenamiento
history = model.fit(
    x_train, y_train,
    epochs=60,                 # límite superior (EarlyStopping suele cortar antes)
    batch_size=64,             # tamaño de lote: equilibrio entre estabilidad y velocidad
    validation_data=(x_val, y_val),
    callbacks=[early, reduce_lr],
    verbose=1
)

# 5) Evaluación del modelo
# Devuelve pérdida y accuracy sobre el conjunto de prueba
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Predicciones por clase (probabilidades) y clase ganadora (argmax)
y_prob = model.predict(x_test, batch_size=256, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

# Reporte detallado por clase (precision/recall/f1/soporte)
print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

# 6) Visualizar resultados — curvas de entrenamiento
hist = history.history  # dict con listas por época (loss, accuracy, val_*)

# Curva de Accuracy (entrenamiento vs validación)
plt.figure(figsize=(6, 5))
plt.plot(hist["accuracy"], label="train_acc")
plt.plot(hist["val_accuracy"], label="val_acc")
plt.xlabel("Época"); plt.ylabel("Accuracy")
plt.title("Curva de Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("cifar10_acc.png", dpi=140)
plt.show()

# Curva de Pérdida (entrenamiento vs validación)
plt.figure(figsize=(6, 5))
plt.plot(hist["loss"], label="train_loss")
plt.plot(hist["val_loss"], label="val_loss")
plt.xlabel("Época"); plt.ylabel("Pérdida (SCCE)")
plt.title("Curva de Pérdida")
plt.legend(); plt.tight_layout()
plt.savefig("cifar10_loss.png", dpi=140)
plt.show()

# 7) Matriz de confusión (qué confunde con qué)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Matriz de Confusión — CIFAR-10")
plt.colorbar()

# Etiquetas de clases cortas (puedes cambiarlas por nombres completos)
classes = ["airp","auto","bird","cat","deer","dog","frog","horse","ship","truck"]
tick = np.arange(len(classes))
plt.xticks(tick, classes, rotation=45, ha="right")
plt.yticks(tick, classes)
plt.xlabel("Predicho"); plt.ylabel("Real")

# Anotar los conteos dentro de cada celda
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("cifar10_cm.png", dpi=140)
plt.show()

# 8) Mostrar algunos ejemplos mal clasificados para inspección visual
errors = np.where(y_pred != y_test)[0]
n_show = min(12, len(errors))  # muestra hasta 12
if n_show > 0:
    rows, cols = 3, 4
    plt.figure(figsize=(12, 9))
    for i in range(n_show):
        idx = errors[i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(x_test[idx])
        plt.title(f"Real={classes[y_test[idx]]}\nPred={classes[y_pred[idx]]}")
        plt.axis("off")
    plt.suptitle("Ejemplos mal clasificados", y=0.98, fontsize=14)
    plt.tight_layout()
    plt.savefig("cifar10_misclassified.png", dpi=140)
    plt.show()
else:
    print("No se encontraron errores en las primeras predicciones (poco probable).")

# ----------------------- Guía: ¿Dónde se aplican las CNN? -----------------------
print("\n=== Dónde se aplican las Redes Neuronales Convolutivas (CNN) ===")
print("- Visión por computadora (general): clasificación, detección (YOLO/SSD), segmentación (U-Net/Mask R-CNN).")
print("- Salud: detección de neumonía en rayos X, lesiones en resonancias, retinopatía diabética.")
print("- Autos autónomos: reconocimiento de peatones, señales, carriles y obstáculos.")
print("- Manufactura: inspección de calidad, detección de defectos, conteo de piezas.")
print("- Agricultura/satélites: clasificación de cultivos, detección de plagas, mapeo de uso de suelo.")
print("- Retail: reconocimiento de productos en anaqueles, análisis de planogramas.")
print("- Seguridad: reidentificación de personas/vehículos, lectura de placas (cumpliendo normativa y privacidad).")
print("- Robótica: visión para agarre y manipulación de objetos.")
