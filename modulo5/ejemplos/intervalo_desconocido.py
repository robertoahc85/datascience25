from scipy.stats import t
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

datos = pd.Series([75, 82, 79, 81, 80, 77, 78, 85, 83, 76])
media = datos.mean()
s = datos.std(ddof=1)
n = len(datos)
t_score = t.ppf(0.975, df=n-1)
error = t_score * (s / np.sqrt(n))

# Visualización
sns.histplot(datos, bins=6)
plt.axvline(media - error, color='red', linestyle='--', label='Límite inferior')
plt.axvline(media + error, color='red', linestyle='--', label='Límite superior')
plt.axvline(media, color='black', label='Media')
plt.title("Intervalo de Confianza para Media (σ desconocida)")
plt.legend()
plt.show()

print(f"IC 95%: [{media - error:.2f}, {media + error:.2f}]")
