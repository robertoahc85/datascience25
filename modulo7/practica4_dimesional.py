# -*- coding: utf-8 -*-
"""
PCA vs t-SNE — Comparación 2D y 3D sobre Digits (scikit-learn)
==============================================================

Este script:
- Carga y estandariza el dataset Digits.
- Aplica PCA (2D, 3D) y t-SNE (2D, 3D).
- Genera 4 gráficas (PNG) y calcula métricas: varianza (PCA), trustworthiness y KNN (5-fold).
- Imprime una tabla resumen e interpretación.

Compatibilidad:
- t-SNE SIN 'n_iter' y con 'learning_rate' numérico (evita errores en scikit-learn antiguos).
"""

# ---------- 1) Imports ----------
import numpy as np                       # Cálculo numérico
import pandas as pd                      # Tabla de resultados
import matplotlib.pyplot as plt          # Gráficas (no estilos extra)
from mpl_toolkits.mplot3d import Axes3D  # Habilita proyección 3D en Matplotlib
import time                              # Medir tiempos

from sklearn.datasets import load_digits                 # Dataset Digits (8x8=64 features)
from sklearn.preprocessing import StandardScaler         # Estandarización (media 0, var 1)
from sklearn.decomposition import PCA                    # PCA (lineal)
from sklearn.manifold import TSNE, trustworthiness       # t-SNE y métrica de vecindad
from sklearn.model_selection import StratifiedKFold, cross_val_score  # Validación cruzada estratificada
from sklearn.neighbors import KNeighborsClassifier       # Clasificador KNN (k=5)

# ---------- 2) Parámetros ----------
SUBMUESTREO = 600               # Usar 600 muestras para acelerar t-SNE; pon None para usar todas
RANDOM_STATE = 42               # Reproducibilidad
TSNE_PERPLEXITY = 20            # Tamaño de vecindario efectivo (5–50)
TSNE_LEARNING_RATE = 200        # Valor numérico (evita 'auto' por compatibilidad)
TSNE_EARLY_EXAGGERATION = 8     # Acelera separación inicial (típico 8–12)

# ---------- 3) Cargar datos + submuestrear + escalar ----------
digits = load_digits()          # Carga (n≈1797, d=64)
X_full, y_full = digits.data, digits.target  # X: features (64), y: dígitos 0..9

if SUBMUESTREO is not None and SUBMUESTREO < len(X_full):
    rng = np.random.RandomState(RANDOM_STATE)         # RNG reproducible
    idx = rng.choice(len(X_full), size=SUBMUESTREO, replace=False)  # Índices aleatorios sin reemplazo
    X, y = X_full[idx], y_full[idx]                   # Subconjunto (p.ej., n=600)
else:
    X, y = X_full, y_full                             # Usa todo el dataset

scaler = StandardScaler()                             # Crea escalador estándar
X_scaled = scaler.fit_transform(X)                    # Ajusta en X y transforma → media 0, var 1

# ---------- 4) PCA 2D ----------
t0 = time.perf_counter()                              # Cronómetro inicio
pca2 = PCA(n_components=2, random_state=RANDOM_STATE) # PCA a 2 componentes (PC1, PC2)
X_pca2 = pca2.fit_transform(X_scaled)                 # Ajusta PCA y proyecta a 2D
pca2_time = time.perf_counter() - t0                  # Tiempo de PCA 2D
pca2_var = pca2.explained_variance_ratio_.sum()       # Varianza acumulada en 2D

# ----- Gráfico PCA 2D -----
plt.figure()                                          # Nueva figura
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=y, s=12, alpha=0.85)  # Colorea por clase
plt.title(f"PCA 2D — Varianza acumulada: {pca2_var:.3f}")       # Título con varianza
plt.xlabel("PC1"); plt.ylabel("PC2")                  # Ejes
plt.tight_layout()                                    # Ajuste de márgenes
plt.savefig("pca_2d.png", dpi=150)                    # Guarda imagen (no mostramos para scripts headless)
# plt.show()                                          # Descomenta si quieres ver en pantalla

# ---------- 5) PCA 3D ----------
t0 = time.perf_counter()                              # Cronómetro inicio
pca3 = PCA(n_components=3, random_state=RANDOM_STATE) # PCA a 3 componentes (PC1, PC2, PC3)
X_pca3 = pca3.fit_transform(X_scaled)                 # Proyección a 3D
pca3_time = time.perf_counter() - t0                  # Tiempo de PCA 3D
pca3_var = pca3.explained_variance_ratio_.sum()       # Varianza acumulada en 3D

# ----- Gráfico PCA 3D -----
fig = plt.figure()                                    # Nueva figura
ax = fig.add_subplot(111, projection='3d')            # Eje 3D
ax.scatter(X_pca3[:, 0], X_pca3[:, 1], X_pca3[:, 2], c=y, s=12, alpha=0.85)  # Scatter 3D
ax.set_title(f"PCA 3D — Varianza acumulada: {pca3_var:.3f}")    # Título con varianza
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3")# Etiquetas ejes
plt.tight_layout()                                    # Ajuste de márgenes
plt.savefig("pca_3d.png", dpi=150)                    # Guarda imagen
# plt.show()

# ---------- 6) t-SNE 2D (compatibilidad: sin n_iter, LR numérico) ----------
t0 = time.perf_counter()                              # Cronómetro inicio
tsne2 = TSNE(
    n_components=2,                                   # Embebido a 2D
    perplexity=TSNE_PERPLEXITY,                       # Vecindario efectivo
    learning_rate=TSNE_LEARNING_RATE,                 # Tasa de aprendizaje (numérico)
    init="pca",                                       # Inicialización PCA (converge más estable)
    early_exaggeration=TSNE_EARLY_EXAGGERATION,       # Exageración inicial
    random_state=RANDOM_STATE,                        # Reproducible
)
X_tsne2 = tsne2.fit_transform(X_scaled)               # Ajusta y transforma
tsne2_time = time.perf_counter() - t0                 # Tiempo t-SNE 2D

# ----- Gráfico t-SNE 2D -----
plt.figure()
plt.scatter(X_tsne2[:, 0], X_tsne2[:, 1], c=y, s=12, alpha=0.85)
plt.title("t-SNE 2D")
plt.xlabel("Dim 1"); plt.ylabel("Dim 2")
plt.tight_layout()
plt.savefig("tsne_2d.png", dpi=150)
# plt.show()

# ---------- 7) t-SNE 3D ----------
t0 = time.perf_counter()                              # Cronómetro inicio
tsne3 = TSNE(
    n_components=3,                                   # Embebido a 3D
    perplexity=TSNE_PERPLEXITY,                       # Vecindario efectivo
    learning_rate=TSNE_LEARNING_RATE,                 # Tasa de aprendizaje
    init="pca",                                       # Inicialización
    early_exaggeration=TSNE_EARLY_EXAGGERATION,       # Exageración inicial
    random_state=RANDOM_STATE,                        # Reproducible
)
X_tsne3 = tsne3.fit_transform(X_scaled)               # Ajusta y transforma
tsne3_time = time.perf_counter() - t0                 # Tiempo t-SNE 3D

# ----- Gráfico t-SNE 3D -----
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne3[:, 0], X_tsne3[:, 1], X_tsne3[:, 2], c=y, s=12, alpha=0.85)
ax.set_title("t-SNE 3D")
ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
plt.tight_layout()
plt.savefig("tsne_3d.png", dpi=150)
# plt.show()

# ---------- 8) Métricas: trustworthiness y KNN ----------
# Trustworthiness: 0..1 (mejor cuanto mayor); mide preservación de vecindarios locales
tw_pca2  = trustworthiness(X_scaled, X_pca2, n_neighbors=5)     # PCA 2D
tw_pca3  = trustworthiness(X_scaled, X_pca3, n_neighbors=5)     # PCA 3D
tw_tsne2 = trustworthiness(X_scaled, X_tsne2, n_neighbors=5)    # t-SNE 2D
tw_tsne3 = trustworthiness(X_scaled, X_tsne3, n_neighbors=5)    # t-SNE 3D

# KNN 5-fold: exactitud como referencia (t-SNE no es para modelado, solo visualización)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)  # Particiones estratificadas
knn = KNeighborsClassifier(n_neighbors=5)                                   # KNN con k=5

acc_orig  = cross_val_score(knn, X_scaled, y, cv=cv, scoring="accuracy")    # 64D original
acc_pca2  = cross_val_score(knn, X_pca2,  y, cv=cv, scoring="accuracy")     # PCA 2D
acc_pca3  = cross_val_score(knn, X_pca3,  y, cv=cv, scoring="accuracy")     # PCA 3D
acc_tsne2 = cross_val_score(knn, X_tsne2, y, cv=cv, scoring="accuracy")     # t-SNE 2D (ilustrativo)
acc_tsne3 = cross_val_score(knn, X_tsne3, y, cv=cv, scoring="accuracy")     # t-SNE 3D (ilustrativo)

# ---------- 9) Tabla resumen ----------
summary = pd.DataFrame({
    "Método": [
        f"Original (64D, n={X_scaled.shape[0]})",  # Sin reducción
        "PCA 2D", "PCA 3D",                        # PCA 2/3D
        "t-SNE 2D", "t-SNE 3D"                     # t-SNE 2/3D
    ],
    "Dimensiones": [X_scaled.shape[1], 2, 3, 2, 3],            # Número de dimensiones del espacio
    "Tiempo_fit (s)": [np.nan, pca2_time, pca3_time, tsne2_time, tsne3_time],  # Tiempos
    "Varianza acumulada (PCA)": [np.nan, pca2_var, pca3_var, np.nan, np.nan],  # Solo PCA
    "Trustworthiness (k=5)": [np.nan, tw_pca2, tw_pca3, tw_tsne2, tw_tsne3],   # Preservación local
    "KNN Acc media (5-fold)": [
        acc_orig.mean(), acc_pca2.mean(), acc_pca3.mean(), acc_tsne2.mean(), acc_tsne3.mean()
    ],
    "KNN Acc std (5-fold)": [
        acc_orig.std(), acc_pca2.std(), acc_pca3.std(), acc_tsne2.std(), acc_tsne3.std()
    ],
})

# ---------- 10) Impresión + orientación ----------
pd.set_option("display.max_columns", None)           # Muestra todas las columnas al imprimir
print("\n=== COMPARACIÓN PCA vs t-SNE — 2D y 3D ===")
print(summary.to_string(index=False))                # Muestra tabla completa

print("\nInterpretación rápida:")
print("- PCA captura varianza global; 3D suele mejorar frente a 2D (ver 'Varianza acumulada').")
print("- t-SNE preserva vecindarios; normalmente alcanza trustworthiness alto, pero tarda más.")
print("- KNN: en 64D (original) suele ser fuerte; PCA-3D > PCA-2D; t-SNE 2D/3D es para visualizar, no para producción.")

print("\nImágenes guardadas:")
print(" - pca_2d.png")
print(" - pca_3d.png")
print(" - tsne_2d.png")
print(" - tsne_3d.png")
