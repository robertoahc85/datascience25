#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Identificando números escritos (Digits de scikit-learn) — versión .py sin Jupyter
=================================================================================

Este script entrena y evalúa un clasificador sobre el dataset "Digits" (8x8)
usando scikit-learn. Funciona 100% desde la terminal (sin notebooks).

Características:
- Carga de datos y partición en train/test
- Escalado de features (StandardScaler)
- Modelo configurable por CLI (MLP o Regresión Logística)
- Métricas: accuracy, classification_report, matriz de confusión (PNG)
- Predicciones de ejemplo guardadas como imagen (PNG)
- Reproducibilidad por random_state
- Código muy comentado, ideal para aprendizaje

Uso rápido:
-----------
# 1) (opcional) crear y activar entorno virtual:
#    python -m venv venv
#    source venv/bin/activate        # macOS/Linux
#    venv\Scripts\activate           # Windows
#
# 2) instalar dependencias:
#    pip install -r requirements.txt
#
# 3) ejecutar con valores por defecto (MLP):
#    python Identificando_numeros_escritos.py
#
# 4) elegir otro modelo o tamaños:
#    python Identificando_numeros_escritos.py --model logistic --test-size 0.25 --random-state 7

"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# -------------------------------
# 1) Utilidades de visualización
# -------------------------------

def guardar_matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: str) -> None:
    """Genera y guarda la matriz de confusión como imagen PNG.

    Args:
        y_true: etiquetas reales
        y_pred: etiquetas predichas
        out_path: ruta de salida .png
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, values_format="d", colorbar=False)
    ax.set_title("Matriz de confusión — Digits")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def guardar_muestra_predicciones(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: str,
    n: int = 16,
) -> None:
    """Guarda una cuadrícula n imágenes con su etiqueta real y predicha.

    Args:
        images: imágenes 8x8 del dataset (shape: [N, 8, 8])
        y_true: etiquetas reales
        y_pred: etiquetas predichas
        out_path: ruta de salida .png
        n: número de ejemplos a mostrar (debe ser cuadrado perfecto: 4, 9, 16, 25, ...)
    """
    n = int(np.clip(n, 4, len(images)))
    lado = int(np.ceil(np.sqrt(n)))  # lado de la cuadrícula
    fig, axes = plt.subplots(lado, lado, figsize=(lado * 2.2, lado * 2.2))
    axes = np.atleast_2d(axes)

    for idx in range(lado * lado):
        ax = axes[idx // lado, idx % lado]
        ax.axis("off")
        if idx < n:
            ax.imshow(images[idx], cmap="gray")
            correcto = "✓" if y_true[idx] == y_pred[idx] else "✗"
            ax.set_title(f"Real:{y_true[idx]} / Pred:{y_pred[idx]} {correcto}", fontsize=8)
    fig.suptitle("Predicciones de ejemplo — Digits", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# -------------------------------
# 2) Carga y partición de datos
# -------------------------------

def cargar_datos(test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Carga el dataset Digits y retorna partición train/test más imágenes.

    Returns:
        X_train, X_test, y_train, y_test, images_test
    """
    digits = load_digits()
    X = digits.data             # (N, 64) features (8x8 -> 64 pixeles)
    y = digits.target           # (N,) etiquetas 0..9
    images = digits.images      # (N, 8, 8) imágenes para visualización

    X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
        X, y, images, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, img_test


# -------------------------------
# 3) Construcción del modelo
# -------------------------------

def construir_pipeline(modelo: str, random_state: int, hidden_layers: List[int]) -> Pipeline:
    """Crea un Pipeline con escalado + clasificador configurable.

    Args:
        modelo: 'mlp' o 'logistic'
        random_state: semilla
        hidden_layers: arquitectura para MLP (ignorada si logistic)

    Returns:
        sklearn.pipeline.Pipeline listo para fit/predict/score
    """
    if modelo == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=random_state,
        )
    elif modelo == "logistic":
        # lbfgs maneja multiclass 'ovr'/'multinomial' automáticamente con buena convergencia
        clf = LogisticRegression(max_iter=2000, n_jobs=None, random_state=random_state)
    else:
        raise ValueError("Modelo no reconocido. Usa 'mlp' o 'logistic'.")

    # Escalar las features suele mejorar el entrenamiento (especialmente MLP / logistic)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipe


# -------------------------------
# 4) Entrenamiento y evaluación
# -------------------------------

def entrenar_y_evaluar(
    pipe: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    img_test: np.ndarray,
    out_dir: str,
) -> None:
    """Entrena el pipeline y guarda métricas e imágenes."""
    print("🧠 Entrenando el modelo...")
    pipe.fit(X_train, y_train)

    print("🔎 Evaluando en test...")
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy en test: {acc:.4f}\n")

    print("📋 Classification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    # Guardar matriz de confusión
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    cm_path = os.path.join(out_dir, "matriz_confusion.png")
    guardar_matriz_confusion(y_test, y_pred, cm_path)
    print(f"🖼  Matriz de confusión guardada en: {cm_path}")

    # Guardar muestra de predicciones
    preds_path = os.path.join(out_dir, "predicciones_ejemplo.png")
    guardar_muestra_predicciones(img_test, y_test, y_pred, preds_path, n=16)
    print(f"🖼  Predicciones de ejemplo guardadas en: {preds_path}")


# -------------------------------
# 5) CLI (argumentos de línea de comando)
# -------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clasificación de dígitos escritos a mano (Digits 8x8) sin Jupyter."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "logistic"],
        help="Modelo a usar: 'mlp' (red neuronal) o 'logistic' (regresión logística multiclase).",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="64,32",
        help="Capas ocultas para MLP, separadas por comas (ej: '128,64,32'). Ignorado si --model=logistic.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporción para test (ej: 0.2 = 20%).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla para reproducibilidad.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="salidas_digits",
        help="Directorio donde se guardarán las imágenes de resultados.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parsear hidden layers a lista de enteros si se usa MLP
    if args.model == "mlp":
        try:
            hidden_layers = [int(x.strip()) for x in args.hidden_layers.split(",") if x.strip()]
            if not hidden_layers:
                raise ValueError
        except Exception:
            raise ValueError("Parámetro --hidden-layers inválido. Ejemplo válido: '128,64,32'")
    else:
        hidden_layers = []

    # Cargar datos
    X_train, X_test, y_train, y_test, img_test = cargar_datos(
        test_size=args.test_size, random_state=args.random_state
    )

    # Construir pipeline
    pipe = construir_pipeline(
        modelo=args.model, random_state=args.random_state, hidden_layers=hidden_layers
    )

    # Entrenar y evaluar
    entrenar_y_evaluar(
        pipe, X_train, y_train, X_test, y_test, img_test, out_dir=args.out_dir
    )

    print("\n🎉 Listo. Revisa la carpeta de salidas y las métricas impresas arriba.")


if __name__ == "__main__":
    main()
