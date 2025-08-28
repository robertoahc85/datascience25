#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DBSCAN vs K-Means sobre make_moons
==================================

Este script muestra cómo aplicar **DBSCAN** (clustering basado en densidad) y compararlo
con **K-Means** usando un dataset sintético `make_moons`.

El objetivo es:
1. Generar datos de dos lunas entrelazadas (no separables linealmente).
2. Aplicar DBSCAN variando el hiperparámetro `eps` y observar el efecto.
3. Comparar visual y cuantitativamente con K-Means (k=2).

─────────────────────────────────────────────────────────────
Requisitos: pip install numpy matplotlib scikit-learn pandas
─────────────────────────────────────────────────────────────
"""

# ===============================
# 1. Importación de librerías
# ===============================
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# scikit-learn: dataset, escalado, clustering y métricas
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


# ===============================
# 2. Configuración global
# ===============================
RANDOM_STATE = 42     # Semilla para reproducibilidad
N_SAMPLES = 600       # Número de puntos (dataset sintético)
NOISE = 0.08          # Nivel de ruido en las lunas
EPS_LIST = [0.10, 0.15, 0.20, 0.25, 0.30]   # Valores de eps a probar en DBSCAN
MIN_SAMPLES = 5       # Número mínimo de vecinos en DBSCAN


# ===============================
# 3. Funciones auxiliares
# ===============================
def generate_data():
    """
    Genera datos sintéticos tipo "dos lunas" (make_moons).

    - Cada punto pertenece a una de dos clases en forma de luna creciente.
    - Agregamos ruido gaussiano para hacerlo más realista.
    - Estandarizamos con StandardScaler para que ambas variables
      estén en la misma escala (media=0, varianza=1).
    """
    X, y_true = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=RANDOM_STATE)

    # Escalamiento: evita que una variable domine las distancias
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    return X_std, y_true


def evaluate_clustering(X, y_true, labels):
    """
    Calcula métricas de calidad de clustering.

    Parámetros:
    - X: datos originales (n×2).
    - y_true: etiquetas verdaderas (solo para evaluación).
    - labels: etiquetas predichas por el algoritmo de clustering.

    Métricas:
    - ARI (Adjusted Rand Index): compara etiquetas predichas y reales.
    - NMI (Normalized Mutual Information): mide acuerdo entre clusters.
    - Silhouette: calidad interna del clustering (compacto y separado).
    """
    ari = adjusted_rand_score(y_true, labels)
    nmi = normalized_mutual_info_score(y_true, labels)

    sil = np.nan  # valor por defecto
    unique = np.unique(labels)
    # silhouette_score solo es válido si hay ≥2 clusters y no todos los puntos en uno solo
    if len(unique) >= 2 and len(unique) < len(labels):
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan

    return ari, nmi, sil


def plot_clusters(X, labels, title="Clusters"):
    """
    Grafica los puntos en 2D coloreados por su etiqueta de cluster.

    - Cada color representa un cluster diferente.
    - Los puntos con etiqueta -1 en DBSCAN se muestran como "ruido".
    - Útil para observar diferencias visuales entre DBSCAN y K-Means.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=25, alpha=0.9, edgecolor="k")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.tight_layout()
    plt.show()


def run_dbscan(X, y_true, eps, min_samples=5, plot=True):
    """
    Ejecuta DBSCAN con parámetros dados.

    - eps: radio de vecindad (distancia máxima para considerar vecinos).
    - min_samples: mínimo de vecinos para que un punto sea núcleo.

    Retorna métricas y (opcionalmente) genera gráfico.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)  # etiquetas (-1 = ruido)

    # Número de clusters detectados (excluyendo ruido)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # Número de puntos marcados como ruido
    n_noise = int(np.sum(labels == -1))

    # Evaluar resultados con métricas
    ari, nmi, sil = evaluate_clustering(X, y_true, labels)

    if plot:
        title = f"DBSCAN | eps={eps}, min_samples={min_samples} | clusters={n_clusters}, ruido={n_noise}"
        plot_clusters(X, labels, title=title)

    return {
        "modelo": f"DBSCAN(eps={eps:.2f}, min_samples={min_samples})",
        "eps": eps,
        "min_samples": min_samples,
        "clusters": n_clusters,
        "ruido": n_noise,
        "ARI": ari,
        "NMI": nmi,
        "Silhouette": sil
    }


def run_kmeans(X, y_true, k=2, plot=True):
    """
    Aplica K-Means como línea base con k=2.

    Limitación teórica:
    - K-Means impone clusters convexos/esféricos.
    - En datos curvos (como make_moons) tiende a cortar incorrectamente.
    """
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)

    # Evaluar con métricas
    ari, nmi, sil = evaluate_clustering(X, y_true, labels)

    if plot:
        title = f"K-Means | k={k} | ARI={ari:.3f}, NMI={nmi:.3f}, Sil={sil:.3f}"
        plot_clusters(X, labels, title=title)

    return {
        "modelo": f"KMeans(k={k})",
        "eps": None,
        "min_samples": None,
        "clusters": len(np.unique(labels)),
        "ruido": 0,
        "ARI": ari,
        "NMI": nmi,
        "Silhouette": sil
    }


# ===============================
# 4. Programa principal
# ===============================
def main():
    # Mostrar versiones (útil para reproducibilidad en prácticas)
    import sklearn
    print(f"Python: {sys.version.split()[0]}")
    print(f"numpy: {np.__version__} | matplotlib: {plt.matplotlib.__version__} | sklearn: {sklearn.__version__}")
    print("=" * 70)

    # 1) Generar datos
    X, y_true = generate_data()
    print(f"Dataset generado: {X.shape[0]} muestras, {X.shape[1]} características\n")

    # 2) Ejecutar DBSCAN variando eps
    resultados = []
    for eps in EPS_LIST:
        res = run_dbscan(X, y_true, eps=eps, min_samples=MIN_SAMPLES, plot=True)
        resultados.append(res)

    # 3) Comparar con K-Means
    res_km = run_kmeans(X, y_true, k=2, plot=True)
    resultados.append(res_km)

    # 4) Mostrar tabla comparativa
    df = pd.DataFrame(resultados)
    print("\n📊 Resultados comparativos DBSCAN vs K-Means:")
    print(df.to_string(index=False))

    # 5) Interpretación didáctica
    print("\n✅ Guía de interpretación:")
    print("- DBSCAN con eps demasiado pequeño → muchos puntos como ruido, clusters fragmentados.")
    print("- DBSCAN con eps demasiado grande → pocos clusters (a veces 1), silhouette baja.")
    print("- K-Means en make_moons suele fallar (corta las lunas en diagonal).")
    print("- DBSCAN bien calibrado captura mejor la forma curva → métricas (ARI/NMI) mayores.")


if __name__ == "__main__":
    main()
