import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os

# Crear carpeta si no existe
os.makedirs("graficas", exist_ok=True)

# Datos base
n = 200
apoyan = 60
p_hat = apoyan / n
z_critical = stats.norm.ppf(0.95)  # 90% bilateral → z = 1.645 aprox

# Calcular intervalo de confianza
margin_error = z_critical * np.sqrt(p_hat * (1 - p_hat) / n)
ci = (p_hat - margin_error, p_hat + margin_error)

# Valores para gráfica
no_apoyan = n - apoyan

# Gráfico de barra en formato vertical
fig, ax = plt.subplots(figsize=(5, 6))

# Barras
ax.bar(0, apoyan, color='green', label=f'Apoyo ({apoyan})')
ax.bar(0, no_apoyan, bottom=apoyan, color='salmon', label=f'No Apoyo ({no_apoyan})')

# Línea de proporción estimada
ax.axhline(p_hat * n, color='blue', linestyle='--', label=f'Estimación: {p_hat:.2f}')
ax.axhline(ci[0] * n, color='red', linestyle='--', label=f'Límite inferior IC: {ci[0]:.2f}')
ax.axhline(ci[1] * n, color='darkblue', linestyle='--', label=f'Límite superior IC: {ci[1]:.2f}')

# Estética
ax.set_ylim(0, n)
ax.set_xticks([0])
ax.set_xticklabels(['Total 200 personas'])
ax.set_ylabel('Cantidad de personas')
ax.set_title('Apoyo a la política con intervalo de confianza del 90%')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

plt.tight_layout()
plt.savefig('graficas/barra_vertical_ic90.png')
plt.show()
