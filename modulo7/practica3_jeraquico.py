#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hierarchical Clustering with SciPy (dendrogram) and scikit-learn (AgglomerativeClustering)
=========================================================================================

Este script implementa paso a paso el **agrupamiento jerárquico** (agglomerative) usando:

1) **SciPy** para construir el *dendrograma* a partir de la **matriz de distancias**.
2) **scikit-learn** para entrenar el modelo `AgglomerativeClustering` y asignar una etiqueta
   de cluster a cada muestra.

Dataset: **Iris** (scikit-learn). Usamos dos variables para facilitar la visualización:
- `sepal length (cm)`
- `sepal width (cm)`

Requisitos didácticos cubiertos
-------------------------------
1. Cargar el conjunto de datos (Iris).
2. Calcular la matriz de distancias y representar los clusters en un dendrograma.
3. Elegir el número de clusters "óptimo" con base en el dendrograma (explicación y ejemplo).
4. Aplicar el algoritmo de clustering jerárquico para asignar cada punto a un cluster.
5. Visualizar los resultados en 2D.

Notas teóricas mínimas
----------------------
- **Clustering jerárquico aglomerativo**: parte con cada muestra como un cluster y va
  fusionándolos recursivamente según una métrica de distancia y un *criterio de enlace*
  (linkage). Ejemplos de *linkage*: `single`, `complete`, `average`, `ward`.
- **Ward** minimiza la varianza intra-cluster en cada fusión y es común con distancia euclídea.
- **Dendrograma**: diagrama en forma de árbol que muestra el historial de fusiones. Cortar
  el dendrograma a cierta altura (umbral de distancia) define el número de clusters.

Sugerencia práctica
-------------------
- Para conjuntos de datos pequeños/medianos, calcular el dendrograma es viable y da intuición.
- Para conjuntos grandes, usar directamente `AgglomerativeClustering` sin dendrograma o
  herramientas como `scikit-learn` + métricas de calidad (p.ej., *silhouette*) para decidir k.

Autor: Tú 🙂
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def load_iris_two_features() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Carga el dataset Iris y devuelve solo dos columnas para visualización 2D.
    Retorna:
        X (np.ndarray): matriz n×2 con [sepal length, sepal width].
        y (np.ndarray): etiquetas verdaderas (0,1,2) para referencia (no se usan para entrenar).
        feature_names (list[str]): nombres de las dos características.
    """
    iris = datasets.load_iris()
    # Índices de columnas: 0=sepal length, 1=sepal width, 2=petal length, 3=petal width
    cols = [0, 1]
    X = iris.data[:, cols]
    y = iris.target
    feature_names = [iris.feature_names[i] for i in cols]
    return X, y, feature_names


def compute_distance_and_linkage(X: np.ndarray, method: str = "ward") -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la matriz de distancias y la matriz de enlace (linkage).

    Parámetros:
        X: datos (n×d).
        method: método de enlace para `scipy.cluster.hierarchy.linkage`.
                - 'ward' requiere distancia euclídea.

    Retorna:
        D (np.ndarray): matriz de distancias cuadrada (n×n).
        Z (np.ndarray): matriz de enlace para construir el dendrograma.
    """
    # Recomendación común: estandarizar para que las características comparen en misma escala
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # pdist computa distancias por pares; squareform la convierte a matriz cuadrada
    dist_vec = pdist(X_std, metric="euclidean")
    D = squareform(dist_vec)

    # linkage: 'ward', 'single', 'complete', 'average', etc.
    Z = linkage(dist_vec, method=method)
    return D, Z


def plot_dendrogram(Z: np.ndarray, truncate_mode: str | None = None, p: int = 12) -> None:
    """
    Dibuja el dendrograma.

    Parámetros:
        Z: matriz de enlace.
        truncate_mode: recortar el dendrograma para datasets grandes (None, 'lastp', etc.).
        p: cuánto mostrar si se usa 'lastp'.
    """
    plt.figure(figsize=(9, 5))
    dendrogram(
        Z,
        truncate_mode=truncate_mode,  # None = completo
        p=p,
        leaf_rotation=90.0,
        leaf_font_size=10.0,
        show_contracted=False,
    )
    plt.title("Dendrograma (linkage='ward')")
    plt.xlabel("Índice de muestra o (cluster)")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.show()


def choose_clusters_from_dendrogram(Z: np.ndarray, maxclust: int | None = 3, distance_threshold: float | None = None) -> np.ndarray:
    """
    Ejemplo de corte del dendrograma para obtener etiquetas de cluster.

    Opción A: especificar `maxclust` (número de clusters deseado).
    Opción B: especificar `distance_threshold` (umbral de distancia al cortar el dendrograma).

    Retorna:
        labels (np.ndarray): etiquetas de cluster (1..k) asignadas por SciPy.
    """
    if distance_threshold is not None and maxclust is not None:
        raise ValueError("Usa solo una de las dos opciones: maxclust o distance_threshold.")

    if distance_threshold is not None:
        labels = fcluster(Z, t=distance_threshold, criterion="distance")
    else:
        # Por defecto devolvemos 3 clusters (Iris tiene 3 especies) para ejemplificar
        labels = fcluster(Z, t=maxclust, criterion="maxclust")
    return labels


def sklearn_agglomerative(X: np.ndarray, n_clusters: int = 3, linkage_method: str = "ward") -> np.ndarray:
    """
    Aplica AgglomerativeClustering de scikit-learn para asignar etiquetas de cluster.

    Nota: `linkage='ward'` requiere `affinity='euclidean'` (es el valor por defecto en versiones recientes).
    """
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_std)
    return labels


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, feature_names: list[str], title: str) -> None:
    """
    Grafica los puntos en 2D coloreados por etiquetas de cluster.

    Parámetros:
        X: datos originales (n×2).
        labels: etiquetas de cluster (np.ndarray).
        feature_names: nombres de las dos características para ejes.
        title: título del gráfico.
    """
    plt.figure(figsize=(6, 5))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, alpha=0.8, edgecolor="k")
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    # 1) Cargar Iris (2 columnas para visualización)
    X, y_true, feature_names = load_iris_two_features()

    # 2) Distancias y dendrograma con SciPy
    D, Z = compute_distance_and_linkage(X, method="ward")
    print("Matriz de distancias (primeras 5×5 celdas):\n", np.round(D[:5, :5], 3))

    # 2b) Dendrograma
    plot_dendrogram(Z, truncate_mode=None)

    # 3) Elegir número de clusters a partir del dendrograma
    #    - Visualmente se busca una “gran brecha” vertical para cortar.
    #    - Para el ejemplo didáctico, usamos k=3 (Iris tiene 3 especies).
    labels_scipy = choose_clusters_from_dendrogram(Z, maxclust=3, distance_threshold=None)

    # 4) Aplicar AgglomerativeClustering (scikit-learn) con k=3
    labels_sklearn = sklearn_agglomerative(X, n_clusters=3, linkage_method="ward")

    # 5) Visualizar resultados
    plot_clusters_2d(X, labels_scipy, feature_names, title="Clusters (SciPy fcluster, k=3)")
    plot_clusters_2d(X, labels_sklearn, feature_names, title="Clusters (sklearn Agglomerative, k=3)")

    # (Opcional) Métrica simple de concordancia entre ambos enfoques (no-invariante a permutación de etiquetas)
    # Nota: Las etiquetas de clustering no están identificadas; podrían diferir en el número pero representar conjuntos similares.
    agreement = np.mean(labels_scipy == labels_sklearn)
    print(f"Acuerdo crudo etiqueta-a-etiqueta entre SciPy y sklearn: {agreement:.3f}")
    print("Recuerda: un re-etiquetado (permutación) podría aumentar el acuerdo si los particionamientos son equivalentes.")


if __name__ == "__main__":
    main()
