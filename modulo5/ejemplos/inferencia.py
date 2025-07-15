import pandas as pd
import matplotlib.pyplot as plt

# Muestra simulada
edades = pd.Series([34, 36, 33, 38, 37, 35, 34, 36, 35, 34])

# Cálculo de la media
media = edades.mean()

# Visualización
plt.hist(edades, bins=5, color='skyblue', edgecolor='black')
plt.axvline(media, color='red', linestyle='--', label=f'Media: {media}')
plt.title("Distribución de edades (muestra)")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.legend()
plt.show()
