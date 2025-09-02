#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RNN (Keras) para Regresión en Boston Housing — Código didáctico con teoría paso a paso
=====================================================================================

Objetivo pedagógico
-------------------
Demostrar el flujo completo de un problema de **regresión** usando **redes neuronales recurrentes** (RNN)
con **Keras (TensorFlow)** sobre el dataset tabular **Boston Housing** (OpenML). Aunque este dataset no es
secuencial por naturaleza, lo **reinterpretamos como una secuencia** de longitud 13 (cada característica es
un "paso de tiempo") con 1 feature por paso, únicamente con fines **didácticos** para practicar RNN/LSTM/GRU.

Mapa del script
---------------
1) Instalación (comentada) y *imports*.
2) Semillas aleatorias para reproducibilidad.
3) Carga de datos desde **OpenML** (porque `load_boston` está deprecado).
4) División en train/valid/test.
5) Escalamiento de variables (StandardScaler).
6) **Reformateo a secuencia**: (n_muestras, 13) -> (n_muestras, 13, 1).
7) Definición del **modelo RNN** (SimpleRNN/LSTM/GRU) + capas densas.
8) Entrenamiento (ajustar épocas y batch_size).
9) Gráficas de desempeño: MSE y MAE (train/val).
10) Predicción en test, cálculo de **RMSE/MAE**, y **caso de prueba**.
11) Guardado de artefactos: CSV de predicciones y PNGs.

Notas importantes
-----------------
- Para datos tabulares, en la práctica se suelen preferir MLPs, árboles (Random Forest, XGBoost) o modelos
  específicos para tabular. Aquí usamos RNN con propósitos formativos.
- La primera descarga de OpenML requiere **internet**; luego queda cacheado en ~/.openml/.
- Si tu GPU está disponible, TensorFlow la usará automáticamente (consulta `tf.config.list_physical_devices('GPU')`).

Requisitos (instalar en tu entorno)
-----------------------------------
    python -m venv .venv
    source .venv/bin/activate            # Windows: .venv\Scripts\activate
    pip install --upgrade pip
    pip install tensorflow scikit-learn matplotlib pandas numpy
"""

# ================================
# 1) IMPORTS CON TEORÍA EN COMENTARIOS
# ================================

# Núcleo científico: manejo numérico y de datos
import os                              # Manejo de rutas/archivos para guardar artefactos
import numpy as np                     # Arreglos y operaciones vectorizadas (base numérica)
import pandas as pd                    # DataFrames: lectura/escritura tabular y utilidades

# Visualización: usaremos matplotlib sin estilos/colores forzados (requisito)
import matplotlib.pyplot as plt        # Gráficas (curvas de entrenamiento, etc.)

# scikit-learn: utilidades clásicas de ML para preparar datos y evaluar
from sklearn.model_selection import train_test_split  # División estratificada/aleatoria
from sklearn.preprocessing import StandardScaler      # Estandarización (media 0, var 1)
from sklearn.metrics import mean_absolute_error, mean_squared_error  # Métricas regresión
from sklearn.datasets import fetch_openml  
from sklearn.metrics import mean_squared_error# Acceso a datasets de OpenML

# TensorFlow/Keras: definición y entrenamiento de redes neuronales
import tensorflow as tf                               # Backend numérico y ejecución acelerada
from tensorflow import keras                          # API de alto nivel (Keras)
from tensorflow.keras import layers                   # Capas (Dense, SimpleRNN, LSTM, GRU, etc.)


# ================================
# 2) PARÁMETROS GLOBALES Y SEMILLAS
# ================================
# Teoría: Para reproducir experimentos, fijar semilla en NumPy y TensorFlow ayuda a repetir resultados (hasta cierto punto).
RANDOM_SEED = 42           # Semilla para reproducibilidad
TEST_SIZE = 0.2            # 20% para prueba final (desempeño fuera de muestra)
VAL_SIZE = 0.2             # 20% del conjunto de entrenamiento se separa como validación
EPOCHS = 150               # Número de pasadas completas por los datos de entrenamiento
BATCH_SIZE = 32            # Tamaño de lote: ejemplos procesados antes de actualizar pesos
RNN_TYPE = "SimpleRNN"     # Opciones: "SimpleRNN", "LSTM", "GRU"
RNN_UNITS = 32             # Número de unidades (dimensión del estado oculto) en la capa recurrente
LEARNING_RATE = 1e-3       # Tasa de aprendizaje para el optimizador Adam

# Fijar semillas
np.random.seed(RANDOM_SEED)             # Semilla para NumPy
tf.random.set_seed(RANDOM_SEED)         # Semilla para TensorFlow (afecta inicialización y dropout)


# ================================
# 3) CARGA DEL DATASET DESDE OPENML
# ================================
# Teoría: Boston Housing contiene 13 características tabulares (ej., crim, rm, lstat) y la variable objetivo 'medv'.
#        `load_boston` fue removido por temas éticos; usamos OpenML ("boston", version=1).
boston = fetch_openml(name="boston", version=1, as_frame=True)  # Descarga/carga como DataFrame
X_df = boston.data                                             # Características (pandas DataFrame)
y_series = boston.target.astype(float)                         # Objetivo en float (Series)

# Información básica del dataset
print("Dimensiones originales:", X_df.shape)                   # (n_muestras, n_features=13)
print("Columnas (features):", list(X_df.columns))              # Nombres de columnas
print("Ejemplo de primeras filas:\n", X_df.head(3))           # Muestra primeras filas
print("Rango objetivo (medv):", (y_series.min(), y_series.max()))  # Mínimo y máximo de medv


# ================================
# 4) DIVISIÓN TRAIN / TEST Y LUEGO VALIDACIÓN
# ================================
# Teoría: Separar TEST aparta datos "nunca vistos" para evaluación honesta. Luego de train, dividimos una fracción para VALID.
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_df.values,                # Convertimos a numpy para eficiencia
    y_series.values,            # Objetivo como arreglo numpy
    test_size=TEST_SIZE,        # 20% para test
    random_state=RANDOM_SEED    # Reproducibilidad
)

# Ahora, de X_train_full generamos un conjunto de validación para ajustar hiperparámetros y monitorear overfitting.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,               # Conjunto de entrenamiento completo
    y_train_full,               # Objetivos correspondientes
    test_size=VAL_SIZE,         # 20% para validación
    random_state=RANDOM_SEED    # Reproducibilidad
)

print("Tamaños -> train:", X_train.shape, "valid:", X_val.shape, "test:", X_test.shape)


# ================================
# 5) ESCALAMIENTO DE CARACTERÍSTICAS
# ================================
# Teoría: Estandarizar (media=0, var=1) acelera y estabiliza el entrenamiento de redes neuronales.
scaler = StandardScaler()               # Instanciamos el estandarizador
X_train_scaled = scaler.fit_transform(X_train)   # Ajuste en train y transformación
X_val_scaled = scaler.transform(X_val)           # Transformación consistente en valid
X_test_scaled = scaler.transform(X_test)         # Transformación consistente en test


# ================================
# 6) REINTERPRETAR TABULAR COMO SECUENCIA PARA RNN
# ================================
# Teoría: Una RNN espera tensores 3D: (batch, timesteps, features). Aquí consideramos 13 "timesteps" y 1 "feature".
def to_sequence(arr_2d: np.ndarray) -> np.ndarray:
    """
    Convierte una matriz 2D (n_muestras, n_features) en 3D (n_muestras, timesteps, features).
    En nuestro caso: timesteps = 13, features = 1.
    """
    # reshape no copia memoria si no es necesario; aquí reinterpreta dimensiones
    return arr_2d.reshape((arr_2d.shape[0], arr_2d.shape[1], 1))

# Aplicamos a cada partición escalada
X_train_seq = to_sequence(X_train_scaled)   # (n_train, 13, 1)
X_val_seq   = to_sequence(X_val_scaled)     # (n_val,   13, 1)
X_test_seq  = to_sequence(X_test_scaled)    # (n_test,  13, 1)

print("Forma secuencial (train):", X_train_seq.shape)  # Confirmación de forma


# ================================
# 7) DEFINICIÓN DEL MODELO RNN (Keras)
# ================================
# Teoría de arquitectura:
# - Capa recurrente (SimpleRNN/LSTM/GRU) "lee" la secuencia y produce una representación (estado oculto).
# - Capa Dense de 16 neuronas con ReLU agrega no linealidad adicional para proyección a espacio intermedio.
# - Capa Dense final (1) produce una salida continua para regresión (sin activación).
def build_rnn_model(input_shape, rnn_type="SimpleRNN", units=32, lr=1e-3) -> keras.Model:
    """
    Construye un modelo secuencial RNN para regresión.
    :param input_shape: tupla (timesteps, features) -> aquí (13, 1)
    :param rnn_type: "SimpleRNN" | "LSTM" | "GRU"
    :param units: tamaño del estado oculto en la capa recurrente
    :param lr: learning rate del optimizador Adam
    """
    model = keras.Sequential(name=f"{rnn_type}_regressor")  # Modelo secuencial

    # Elegimos el tipo de celda recurrente: SimpleRNN, LSTM o GRU.
    if rnn_type == "LSTM":
        model.add(layers.LSTM(units, input_shape=input_shape))  # LSTM maneja dependencias largas con compuertas
    elif rnn_type == "GRU":
        model.add(layers.GRU(units, input_shape=input_shape))   # GRU es más ligera que LSTM, buen rendimiento
    else:
        model.add(layers.SimpleRNN(units, input_shape=input_shape))  # SimpleRNN: versión básica

    model.add(layers.Dense(16, activation="relu"))     # Capa oculta densa para mayor capacidad de representación
    model.add(layers.Dense(1, activation="linear"))    # Salida lineal (regresión)

    # Compilación: MSE como función de pérdida y MAE como métrica comprensible (error absoluto medio)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),  # Adam adapta la tasa de aprendizaje por parámetro
        loss="mse",                                        # Minimizar Error Cuadrático Medio
        metrics=["mae"]                                    # Reportar Error Absoluto Medio durante el entrenamiento
    )
    return model

# Construimos el modelo con la forma de entrada secuencial
model = build_rnn_model(input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                        rnn_type=RNN_TYPE, units=RNN_UNITS, lr=LEARNING_RATE)

# Resumen del modelo: útil para verificar número de parámetros y arquitectura
model.summary()


# ================================
# 8) ENTRENAMIENTO DEL MODELO
# ================================
# Teoría: Entrenar = iterar (épocas) sobre lotes (batch_size) ajustando pesos para minimizar la pérdida en train,
#         monitoreando valid para detectar sobreajuste.
history = model.fit(
    X_train_seq, y_train,                 # Datos de entrenamiento
    validation_data=(X_val_seq, y_val),   # Datos de validación
    epochs=EPOCHS,                        # Épocas totales
    batch_size=BATCH_SIZE,                # Tamaño de lote
    verbose=1                             # 1 = barra de progreso + métricas por época
)


# ================================
# 9) VISUALIZACIÓN DEL RENDIMIENTO
# ================================
# Teoría: Observar curvas de pérdida (MSE) y MAE en train/val ayuda a diagnosticar underfitting/overfitting.

# Curva de pérdida (MSE)
fig1 = plt.figure()                                       # Crear figura nueva
plt.plot(history.history["loss"], label="loss (train)")   # Pérdida en entrenamiento
plt.plot(history.history["val_loss"], label="loss (val)") # Pérdida en validación
plt.title("Evolución de la pérdida (MSE)")                # Título
plt.xlabel("Época")                                       # Eje X
plt.ylabel("MSE")                                         # Eje Y
plt.legend()                                              # Leyenda
plt.tight_layout()                                        # Mejor espaciado
fig1_path = "grafica_loss.png"                            # Ruta de guardado
fig1.savefig(fig1_path, dpi=120)                          # Guardar PNG
print(f"Guardado: {fig1_path}")                           # Confirmación por consola

# Curva de MAE
fig2 = plt.figure()                                       # Nueva figura
plt.plot(history.history["mae"], label="mae (train)")     # MAE entrenamiento
plt.plot(history.history["val_mae"], label="mae (val)")   # MAE validación
plt.title("Evolución del MAE")                            # Título
plt.xlabel("Época")                                       # Eje X
plt.ylabel("MAE")                                         # Eje Y
plt.legend()                                              # Leyenda
plt.tight_layout()                                        # Mejor espaciado
fig2_path = "grafica_mae.png"                             # Ruta de guardado
fig2.savefig(fig2_path, dpi=120)                          # Guardar PNG
print(f"Guardado: {fig2_path}")                           # Confirmación

# Nota: para visualizar en notebooks, llamar a plt.show(). Aquí solo guardamos PNGs.


# ================================
# 10) EVALUACIÓN Y PREDICCIONES
# ================================
# Teoría: MAE y RMSE resumen el error medio absoluto y la raíz del error cuadrático medio, respectivamente.
#        Menores valores indican mejor ajuste (en la escala de la variable objetivo).
y_pred = model.predict(X_test_seq).ravel()                # Predicciones (vector 1D)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE = sqrt(MSE)
mae  = mean_absolute_error(y_test, y_pred)                # MAE

print(f"RMSE (test): {rmse:.3f}")                         # Mostrar RMSE con 3 decimales
print(f"MAE  (test): {mae:.3f}")                          # Mostrar MAE  con 3 decimales

# Caso de prueba: mostrar un ejemplo concreto
idx = 0                                                  # Índice del ejemplo en X_test
print("\n--- Caso de prueba ---")                       # Encabezado
print("Entrada original (13 features, primeras 5):")     # Texto de apoyo
print(X_test[idx][:5])                                   # Mostrar las 5 primeras características reales (sin escalar)
print(f"Valor real (medv): {y_test[idx]:.2f}")           # Valor objetivo real del ejemplo
print(f"Predicción       : {y_pred[idx]::.2f}")          # Predicción del modelo para ese ejemplo

# ================================
# 11) GUARDAR RESULTADOS
# ================================
# Teoría: Guardar artefactos facilita trazabilidad y análisis posterior.
results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})  # DataFrame con pares real-predicho
results_path = "predicciones_test.csv"                            # Nombre de archivo
results_df.to_csv(results_path, index=False)                      # Exportar a CSV
print(f"Guardado: {results_path}")                                # Confirmación

# Fin del script.
# Sugerencias de experimentación:
# - Cambiar RNN_TYPE a "LSTM" o "GRU".
# - Ajustar RNN_UNITS, EPOCHS, BATCH_SIZE, LEARNING_RATE.
# - Probar capas recurrentes apiladas (return_sequences=True en la primera) + GlobalAveragePooling1D.
# - Comparar contra un MLP para ver diferencias en desempeño con tabulares.
