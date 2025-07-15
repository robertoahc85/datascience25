import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


poblacion = np.random.normal(50, 10, 10000)
medias = [np.mean(np.random.choice(poblacion, 30)) for _ in range(100)]
df_medias = pd.DataFrame({'media_muestral': medias})

# Visualización
sns.histplot(df_medias['media_muestral'], kde=True, bins=20)
plt.title("Distribución de Medias Muestrales")
plt.xlabel("Media")
plt.ylabel("Frecuencia")
plt.axvline(df_medias['media_muestral'].mean(), color='red', linestyle='--', label='Media')
plt.legend()
plt.show()

proporciones = [np.mean(np.random.binomial(1, 0.6, 40)) for _ in range(100)]
df_prop = pd.DataFrame({'proporcion_muestral': proporciones})

sns.histplot(df_prop['proporcion_muestral'], kde=True, bins=10)
plt.title("Distribución de Proporciones Muestrales")
plt.xlabel("Proporción")
plt.ylabel("Frecuencia")
plt.axvline(df_prop['proporcion_muestral'].mean(), color='green', linestyle='--')
plt.show()
