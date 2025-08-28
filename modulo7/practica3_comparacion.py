#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ============================================================
# SEGMENTACIÓN DE CLIENTES: K-MEANS vs JERÁRQUICO vs DBSCAN
# Objetivo: Agrupar 100 clientes según Monto (MXN) y Frecuencia (visitas/mes).
# Requisitos:
#   pip install numpy matplotlib scikit-learn scipy
# ============================================================

# -----------------------------
# Importación de bibliotecas
# -----------------------------
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

# K-Means y métricas
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, confusion_matrix

# Escalamiento
from sklearn.preprocessing import StandardScaler

# Jerárquico (SciPy para dendrograma)
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# -----------------------------
# 1) Configuración y reproducibilidad
# -----------------------------
np.random.seed(42)

# -----------------------------
# 2) Generación de datos simulados (100 clientes, 3 segmentos)
# -----------------------------
# Segmento A: Bajo gasto / baja frecuencia
n_a = 40
monto_a = np.random.normal(loc=400, scale=120, size=n_a)
freq_a  = np.random.normal(loc=3,   scale=1.2,  size=n_a)

# Segmento B: Gasto medio / frecuencia media
n_b = 35
monto_b = np.random.normal(loc=1200, scale=250, size=n_b)
freq_b  = np.random.normal(loc=12,   scale=3,   size=n_b)

# Segmento C: Alto gasto / alta frecuencia
n_c = 25
monto_c = np.random.normal(loc=2200, scale=300, size=n_c)
freq_c  = np.random.normal(loc=24,   scale=4,   size=n_c)

# Unimos segmentos
monto = np.concatenate([monto_a, monto_b, monto_c])
freq  = np.concatenate([freq_a,  freq_b,  freq_c])

# Rango realista
monto = np.clip(monto, 100, 5000)
freq  = np.clip(freq, 1, 50)

# Matriz X y mezcla aleatoria de filas
X = np.column_stack([monto, freq])
idx = np.random.permutation(X.shape[0])
X = X[idx, :]

# -----------------------------
# 3) Estandarización
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# 4) K-MEANS: selección de K, entrenamiento y visualización
# ============================================================
ks = range(2, 8)
inertias = []
silhouettes = []

for k in ks:
    km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42)
    km_tmp.fit(X_scaled)
    inertias.append(km_tmp.inertia_)
    labels_k = km_tmp.labels_
    sil_k = silhouette_score(X_scaled, labels_k)
    silhouettes.append(sil_k)

# Método del codo
plt.figure(figsize=(5, 4))
plt.plot(ks, inertias, marker='o')
plt.title("Método del Codo (K-Means)")
plt.xlabel("K")
plt.ylabel("Inercia (SSE)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# Silhouette vs K
plt.figure(figsize=(5, 4))
plt.plot(ks, silhouettes, marker='o')
plt.title("Coeficiente de Silueta (K-Means)")
plt.xlabel("K")
plt.ylabel("Silhouette")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# Entrenamiento final con K=3 (acorde a la simulación)
k = 3
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
kmeans.fit(X_scaled)

labels_km = kmeans.labels_
centroids_scaled = kmeans.cluster_centers_
sse = kmeans.inertia_
sil_km = silhouette_score(X_scaled, labels_km)

print(f"[K-Means] K={k} | SSE={sse:.2f} | Silhouette={sil_km:.3f}")
print("Distribución (K-Means):", Counter(labels_km))

centroids_orig = scaler.inverse_transform(centroids_scaled)
for i, c in enumerate(centroids_orig):
    print(f"Centroide KM {i}: Monto≈{c[0]:.0f} MXN, Frec≈{c[1]:.1f}/mes")

# Plot K-Means
plt.figure(figsize=(6, 5))
colors = ["tab:blue", "tab:orange", "tab:green"]
for i in range(k):
    plt.scatter(X[labels_km == i, 0], X[labels_km == i, 1], s=35, alpha=0.85, c=colors[i], label=f"Cluster {i}")
plt.scatter(centroids_orig[:, 0], centroids_orig[:, 1], marker="X", s=200, c="black", label="Centroides")
plt.title("Segmentación con K-Means (K=3)")
plt.xlabel("Monto (MXN)")
plt.ylabel("Frecuencia (visitas/mes)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()

# ============================================================
# 5) JERÁRQUICO: Dendrograma (SciPy) + Agglomerative (sklearn)
# ============================================================
# Dendrograma con linkage='ward'
dist_vec = pdist(X_scaled, metric="euclidean")
Z = linkage(dist_vec, method="ward")

plt.figure(figsize=(9, 5))
dendrogram(Z, leaf_rotation=90.0, leaf_font_size=9.0, show_contracted=False)
plt.title("Dendrograma (Jerárquico, linkage='ward')")
plt.xlabel("Índice de muestra / (cluster)")
plt.ylabel("Distancia")
plt.tight_layout()
plt.show()

# Cortamos el dendrograma en k=3 (etiquetas 1..k)
labels_h_scipy = fcluster(Z, t=3, criterion="maxclust")

# Silhouette jerárquico (SciPy)
sil_h_scipy = silhouette_score(X_scaled, labels_h_scipy)
print(f"[Jerárquico SciPy] K=3 | Silhouette={sil_h_scipy:.3f}")
print("Distribución (Jerárquico SciPy):", Counter(labels_h_scipy))

# Plot Jerárquico (SciPy)
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels_h_scipy, s=35, alpha=0.85, edgecolor="k", linewidth=0.5)
plt.title("Clusters (Jerárquico SciPy, K=3)")
plt.xlabel("Monto (MXN)")
plt.ylabel("Frecuencia (visitas/mes)")
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()

# AgglomerativeClustering (sklearn) con linkage='ward'
agg = AgglomerativeClustering(n_clusters=3, linkage="ward")
labels_h_sk = agg.fit_predict(X_scaled)
sil_h_sk = silhouette_score(X_scaled, labels_h_sk)
print(f"[Jerárquico sklearn] K=3 | Silhouette={sil_h_sk:.3f}")
print("Distribución (Jerárquico sklearn):", Counter(labels_h_sk))

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=labels_h_sk, s=35, alpha=0.85, edgecolor="k", linewidth=0.5)
plt.title("Clusters (Jerárquico sklearn, K=3)")
plt.xlabel("Monto (MXN)")
plt.ylabel("Frecuencia (visitas/mes)")
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()

# Comparación básica K-Means vs Jerárquico (tabla de contingencia)
cm_km_h = confusion_matrix(labels_km, labels_h_scipy - 1)  # fcluster es 1..k; paso a 0..k-1
print("\nTabla de contingencia (filas=KMeans, columnas=Jerárquico SciPy):\n", cm_km_h)

# ============================================================
# 6) DBSCAN: barrido simple de eps y visualización
# ============================================================
# Nota: DBSCAN no requiere K; depende de eps (radio) y min_samples.
# Elegimos un eps inicial razonable y probamos varios valores.
eps_values = [0.3, 0.4, 0.5, 0.6]
min_samples = 5

mejor_sil = -1
mejor_eps = None
mejor_labels_db = None

print("\n===== DBSCAN: prueba de eps =====")
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_db = db.fit_predict(X_scaled)

    # DBSCAN asigna ruido como -1. Para silhouette, necesitamos ≥ 2 clusters reales.
    mask = labels_db != -1
    unique_clusters = np.unique(labels_db[mask])
    if len(unique_clusters) >= 2 and mask.sum() > 0:
        sil_db = silhouette_score(X_scaled[mask], labels_db[mask])
        print(f"eps={eps:.2f} | clusters={unique_clusters.size} (sin ruido) | silhouette={sil_db:.3f} | ruido={np.sum(labels_db==-1)}")
        if sil_db > mejor_sil:
            mejor_sil = sil_db
            mejor_eps = eps
            mejor_labels_db = labels_db.copy()
    else:
        print(f"eps={eps:.2f} | No se puede calcular silhouette (clusters válidos < 2). | ruido={np.sum(labels_db==-1)}")

# Entrenamiento final DBSCAN con mejor eps encontrado (si lo hubo)
if mejor_labels_db is not None:
    print(f"\n[DBSCAN] Mejor eps={mejor_eps:.2f} | Silhouette={mejor_sil:.3f}")
    labels_db = mejor_labels_db
else:
    # fallback: tomar el último ajuste para mostrar gráfico aunque no haya silhouette
    db = DBSCAN(eps=eps_values[-1], min_samples=min_samples)
    labels_db = db.fit_predict(X_scaled)
    print("\n[DBSCAN] No se obtuvo silhouette válido; se muestran clusters con eps último por referencia.")

# Distribución DBSCAN (incluye ruido -1)
print("Distribución (DBSCAN):", Counter(labels_db))

# Plot DBSCAN
plt.figure(figsize=(6, 5))
unique = np.unique(labels_db)
for lab in unique:
    mask = labels_db == lab
    if lab == -1:
        # Ruido en negro con marcador 'x'
        plt.scatter(X[mask, 0], X[mask, 1], s=55, marker='x', label="Ruido (-1)", c="black")
    else:
        plt.scatter(X[mask, 0], X[mask, 1], s=35, alpha=0.85, label=f"Cluster {lab}")
plt.title(f"Clusters con DBSCAN (eps={mejor_eps if mejor_eps is not None else eps_values[-1]:.2f}, min_samples={min_samples})")
plt.xlabel("Monto (MXN)")
plt.ylabel("Frecuencia (visitas/mes)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.35)
plt.tight_layout()
plt.show()

# ============================================================
# 7) Resumen comparativo de métricas
# ============================================================
print("\n===== RESUMEN MÉTRICAS (Silhouette) =====")
print(f"K-Means (K=3):           {sil_km:.3f}")
print(f"Jerárquico SciPy (K=3):  {sil_h_scipy:.3f}")
print(f"Jerárquico sklearn (K=3):{sil_h_sk:.3f}")

# Silhouette DBSCAN se reporta “sin ruido” si se obtuvo
if mejor_labels_db is not None:
    print(f"DBSCAN (mejor eps={mejor_eps:.2f}): {mejor_sil:.3f} (solo puntos no-ruido)")
else:
    print("DBSCAN: sin silhouette válido (clusterización insuficiente o ruido excesivo).")

print("\nNotas:")
print("- K-Means requiere K; útil cuando se conocen segmentos esperados.")
print("- Jerárquico ofrece dendrograma para decidir K y entender fusiones.")
print("- DBSCAN detecta formas arbitrarias y ruido; no requiere K pero depende de eps/min_samples.")
