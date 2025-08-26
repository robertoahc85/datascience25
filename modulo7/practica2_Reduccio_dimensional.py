# jerarquico_iris.py
# ------------------------------------------------------------
# Implementando el Agrupamiento Jerárquico en el dataset Iris
# - Dendrograma con scipy
# - Clustering con scikit-learn (AgglomerativeClustering)
# - Visualización en 2D (sepal length vs sepal width)
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform

# -----------------------------
# 1) Cargar datos (Iris)
# -----------------------------
iris = datasets.load_iris()
X_full = iris.data  # columnas: [sepal length, sepal width, petal length, petal width]
y = iris.target     # etiquetas reales (solo para referencia)
feature_names = iris.feature_names

# Usaremos solo 2 características para visualizar mejor
# sepal length (cm) -> columna 0
# sepal width  (cm) -> columna 1
X = X_full[:, [0, 1]]

# (Opcional) Estandarizar para que ambas variables estén en la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 2) Matriz de distancias y dendrograma
# -----------------------------
# Calculamos distancias euclidianas entre pares
dist_matrix = pdist(X_scaled, metric='euclidean')

# 'linkage' calcula el agrupamiento jerárquico (método "ward" minimiza varianza intra-cluster)
Z = linkage(dist_matrix, method='ward')

# Visualizamos el dendrograma
plt.figure(figsize=(12, 5))
plt.title("Dendrograma (Iris, 2 features: sepal length & sepal width)")
plt.xlabel("Muestras")
plt.ylabel("Distancia (Ward)")
dendro = dendrogram(Z, no_labels=True, color_threshold=None)
# Tip: Puedes dibujar una línea horizontal para sugerir un corte visual:
# plt.axhline(y=8, color='r', linestyle='--', label='Corte sugerido')
# plt.legend()
plt.tight_layout()
plt.show()

# -----------------------------
# 3) Elegir número de clusters
# -----------------------------
# Normalmente se decide visualmente con el dendrograma.
# Para facilitar el script, mostraremos dos enfoques:
# A) Fijar explícitamente n_clusters basándonos en lo típico de Iris (3)
n_clusters = 3

# B) (Opcional) Derivar etiquetas con un umbral de distancia usando fcluster (Scipy)
#    - Ajusta 't' (threshold) hasta ver ~3 grupos
# labels_from_dendro = fcluster(Z, t=8, criterion='distance')
# print("Clusters (fcluster) encontrados:", len(np.unique(labels_from_dendro)))

# -----------------------------
# 4) Clustering jerárquico con scikit-learn
# -----------------------------
# El modelo de sklearn NO usa Z directamente: recalcula el enlace internamente.
# Usamos el mismo criterio: linkage='ward' y affinity='euclidean'
agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
labels = agg.fit_predict(X_scaled)

# -----------------------------
# 5) Visualización de resultados
# -----------------------------
plt.figure(figsize=(7, 6))
plt.title(f"Agrupamiento Jerárquico (n_clusters={n_clusters})")
plt.scatter(X[:, 0], X[:, 1], c=labels, s=60)
plt.xlabel(feature_names[0])  # sepal length (cm)
plt.ylabel(feature_names[1])  # sepal width  (cm)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Métrica de calidad (silhouette) solo informativa
sil = silhouette_score(X_scaled, labels, metric='euclidean')
print(f"Silhouette score (euclídea, escalado): {sil:.3f}")

# -----------------------------
# 6) Extra: probar otros métodos de enlace
# -----------------------------
# Puedes comparar rápidamente distintos 'linkage' y ver su impacto
for method in ['ward', 'average', 'complete', 'single']:
    if method == 'ward':
        agg_m = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    else:
        # Para average/complete/single se puede usar 'metric="euclidean"'
        agg_m = AgglomerativeClustering(n_clusters=n_clusters, linkage=method, metric='euclidean')

    labels_m = agg_m.fit_predict(X_scaled)
    sil_m = silhouette_score(X_scaled, labels_m)
    print(f"[Comparación] linkage={method:<8} -> silhouette={sil_m:.3f}")
