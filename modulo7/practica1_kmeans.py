# ============================================================
# SEGMENTACIÓN DE CLIENTES CON K-MEANS
# Objetivo: Agrupar 100 clientes según dos variables: Monto gastado (MXN) y Frecuencia de compra (visitas/mes).
# Requisitos: Instalar las bibliotecas necesarias mediante el comando:
#   pip install numpy matplotlib scikit-learn
# ============================================================

# Importación de bibliotecas necesarias
import numpy as np  # Para operaciones numéricas y manejo de arreglos
import matplotlib.pyplot as plt  # Para visualización de datos
from sklearn.cluster import KMeans  # Algoritmo K-Means para clustering
from sklearn.metrics import silhouette_score  # Métrica para evaluar calidad de los clusters
from sklearn.preprocessing import StandardScaler  # Para escalar datos
from collections import Counter  # Para contar la distribución de clientes por cluster

# -----------------------------
# 1) Configuración y reproducibilidad
# -----------------------------
# Establecemos una semilla para garantizar que los resultados sean reproducibles.
# Esto asegura que los números aleatorios generados sean consistentes en cada ejecución.
np.random.seed(42)

# -----------------------------
# 2) Generación de datos simulados
#    - Creamos un conjunto de datos sintético con 100 clientes.
#    - Cada cliente tiene dos características: Monto gastado (MXN) y Frecuencia de compra (visitas/mes).
#    - Simulamos tres segmentos de clientes con distribuciones normales para reflejar comportamientos realistas:
#      a) Segmento A: Clientes con bajo gasto y baja frecuencia.
#      b) Segmento B: Clientes con gasto y frecuencia medios.
#      c) Segmento C: Clientes con alto gasto y alta frecuencia (clientes "VIP").
# -----------------------------

# Segmento A: Bajo gasto / baja frecuencia
n_a = 40  # Número de clientes en este segmento
monto_a = np.random.normal(loc=400, scale=120, size=n_a)  # Monto promedio ~400 MXN, desviación estándar 120
freq_a = np.random.normal(loc=3, scale=1.2, size=n_a)     # Frecuencia promedio ~3 visitas/mes, desviación estándar 1.2

# Segmento B: Gasto medio / frecuencia media
n_b = 35  # Número de clientes en este segmento
monto_b = np.random.normal(loc=1200, scale=250, size=n_b)  # Monto promedio ~1200 MXN, desviación estándar 250
freq_b = np.random.normal(loc=12, scale=3, size=n_b)       # Frecuencia promedio ~12 visitas/mes, desviación estándar 3

# Segmento C: Alto gasto / alta frecuencia (clientes "VIP")
n_c = 25  # Número de clientes en este segmento
monto_c = np.random.normal(loc=2200, scale=300, size=n_c)  # Monto promedio ~2200 MXN, desviación estándar 300
freq_c = np.random.normal(loc=24, scale=4, size=n_c)       # Frecuencia promedio ~24 visitas/mes, desviación estándar 4

# Concatenamos los datos de los tres segmentos para formar un conjunto completo
monto = np.concatenate([monto_a, monto_b, monto_c])  # Vector de montos (100 clientes)
freq = np.concatenate([freq_a, freq_b, freq_c])      # Vector de frecuencias (100 clientes)

# Aseguramos que los valores estén dentro de rangos realistas
# - Monto: mínimo 100 MXN, máximo 5000 MXN
# - Frecuencia: mínimo 1 visita/mes, máximo 50 visitas/mes
monto = np.clip(monto, 100, 5000)
freq = np.clip(freq, 1, 50)

# Creamos la matriz de datos X (100 filas, 2 columnas: monto y frecuencia)
X = np.column_stack([monto, freq])

# Mezclamos aleatoriamente las filas de X para evitar sesgos en el orden
# Esto simula un conjunto de datos real donde los segmentos no están preordenados
idx = np.random.permutation(X.shape[0])
X = X[idx, :]

# -----------------------------
# 3) Escalamiento de los datos
#    - K-Means es sensible a las diferencias en las escalas de las variables.
#    - Usamos StandardScaler para estandarizar los datos (media=0, desviación estándar=1).
#    - Esto asegura que ambas características (monto y frecuencia) tengan el mismo peso en el clustering.
# -----------------------------
scaler = StandardScaler()  # Inicializamos el escalador
X_scaled = scaler.fit_transform(X)  # Ajustamos y transformamos los datos

# -----------------------------
# 4) Selección del número óptimo de clusters (K)
#    - Evaluamos diferentes valores de K (número de clusters) utilizando:
#      a) Método del codo: Minimiza la inercia (suma de distancias cuadradas a los centroides).
#      b) Coeficiente de silueta: Mide la cohesión dentro de los clusters y la separación entre ellos.
#    - Probamos K entre 2 y 7 para encontrar un balance entre simplicidad y calidad.
# -----------------------------
ks = range(2, 8)  # Rango de valores de K a evaluar
inertias = []     # Lista para almacenar la inercia de cada K
silhouettes = []  # Lista para almacenar el coeficiente de silueta de cada K

for k in ks:
    # Entrenamos el modelo K-Means con el valor actual de K
    km = KMeans(n_clusters=k, n_init=10, random_state=42)  # n_init=10 asegura múltiples inicializaciones para mayor estabilidad
    km.fit(X_scaled)  # Ajustamos el modelo a los datos escalados
    inertias.append(km.inertia_)  # Guardamos la inercia (suma de distancias cuadradas)
    labels_k = km.labels_  # Etiquetas de los clusters asignados
    sil = silhouette_score(X_scaled, labels_k)  # Calculamos el coeficiente de silueta
    silhouettes.append(sil)

# Visualización del método del codo
plt.figure(figsize=(5, 4))
plt.plot(ks, inertias, marker='o', color='tab:blue')
plt.title("Método del Codo para Selección de K")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Inercia (Suma de Distancias Cuadradas)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# Visualización del coeficiente de silueta
plt.figure(figsize=(5, 4))
plt.plot(ks, silhouettes, marker='o', color='tab:orange')
plt.title("Coeficiente de Silueta para Selección de K")
plt.xlabel("Número de Clusters (K)")
plt.ylabel("Coeficiente de Silueta")
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()

# -----------------------------
# 5) Entrenamiento del modelo K-Means con K=3
#    - Elegimos K=3 basados en la simulación inicial (tres segmentos) y los resultados del método del codo/silueta.
#    - Ajustamos el modelo final y extraemos métricas clave.
# -----------------------------
k = 3  # Número de clusters seleccionado
kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Inicializamos el modelo
kmeans.fit(X_scaled)  # Ajustamos el modelo a los datos escalados

# Extraemos resultados
labels = kmeans.labels_  # Etiqueta de cluster asignada a cada cliente
centroids_scaled = kmeans.cluster_centers_  # Coordenadas de los centroides en el espacio escalado
sse = kmeans.inertia_  # Inercia final del modelo
sil = silhouette_score(X_scaled, labels)  # Coeficiente de silueta del modelo

# Imprimimos métricas del modelo
print(f"Número de Clusters: {k}")
print(f"Inercia (Suma de Distancias Cuadradas): {sse:.2f}")
print(f"Coeficiente de Silueta: {sil:.3f}")
print("Distribución de Clientes por Cluster:", Counter(labels))

# -----------------------------
# 6) Análisis de los centroides en la escala original
#    - Transformamos los centroides escalados a la escala original para interpretación.
#    - Esto nos permite entender los valores promedio de monto y frecuencia por cluster.
# -----------------------------
centroids_original = scaler.inverse_transform(centroids_scaled)  # Revertimos el escalamiento

# Imprimimos los centroides en la escala original
for i, c in enumerate(centroids_original):
    print(f"Centroide {i}: Monto ≈ {c[0]:.0f} MXN, Frecuencia ≈ {c[1]:.1f} visitas/mes")

# -----------------------------
# 7) Visualización de los clusters
#    - Graficamos los clientes en un gráfico de dispersión (monto vs. frecuencia).
#    - Cada cluster se representa con un color diferente, y los centroides se marcan con una "X".
# -----------------------------
plt.figure(figsize=(6, 5))
colors = ["tab:blue", "tab:orange", "tab:green"]  # Colores para los clusters

# Graficamos los puntos de cada cluster
for i in range(k):
    plt.scatter(
        X[labels == i, 0],  # Monto de los clientes en el cluster i
        X[labels == i, 1],  # Frecuencia de los clientes en el cluster i
        s=35,               # Tamaño de los puntos
        alpha=0.8,          # Transparencia
        c=colors[i],        # Color del cluster
        label=f"Cluster {i}"
    )

# Graficamos los centroides en la escala original
plt.scatter(
    centroids_original[:, 0],  # Monto de los centroides
    centroids_original[:, 1],  # Frecuencia de los centroides
    marker="X",                # Símbolo para los centroides
    s=200,                     # Tamaño del símbolo
    c="black",                 # Color de los centroides
    label="Centroides"
)

plt.title("Segmentación de Clientes con K-Means")
plt.xlabel("Monto Gastado (MXN)")
plt.ylabel("Frecuencia de Compra (visitas/mes)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.35)
plt.show()

# -----------------------------
# 8) Interpretación de los clusters
#    - Ordenamos los clusters por monto promedio para asignar etiquetas descriptivas.
#    - Esto facilita la interpretación de los segmentos (bajo, medio, alto).
# -----------------------------
orden = np.argsort(centroids_original[:, 0])  # Índices de los centroides ordenados por monto
mapa_etiquetas = {
    orden[0]: "Bajo gasto / baja-moderada frecuencia",
    orden[1]: "Gasto medio / frecuencia media",
    orden[2]: "Alto gasto / alta frecuencia"
}

# Imprimimos la interpretación de cada cluster
print("\nInterpretación sugerida por centroide:")
for i in range(k):
    c = centroids_original[i]
    print(f"- Cluster {i} → {mapa_etiquetas[i]} "
          f"(Monto ≈ {c[0]:.0f} MXN, Frecuencia ≈ {c[1]:.1f} visitas/mes)")

# -----------------------------
# 9) Asignación de etiquetas legibles a los clientes
#    - Asignamos una etiqueta descriptiva a cada cliente según su cluster.
#    - Mostramos un ejemplo con los primeros 10 clientes.
# -----------------------------
etiquetas_legibles = np.array([mapa_etiquetas[l] for l in labels])  # Etiquetas descriptivas para cada cliente

# Imprimimos los primeros 10 clientes con sus datos y etiquetas
print("\nPrimeros 10 clientes (Monto, Frecuencia, Cluster):")
for i in range(10):
    print(f"Cliente {i:02d} -> (Monto = {X[i,0]:.0f} MXN, Frecuencia = {X[i,1]:.0f} visitas/mes) | {etiquetas_legibles[i]}")