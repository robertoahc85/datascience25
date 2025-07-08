#  ¿La música de fondo mejora la memoria a corto plazo?

# Problema Estudiantes suelen poner música mientras estudian y afirman recordar mejor.

# Hipótesis nula (H₀) Escuchar música instrumental no cambia las puntuaciones de memoria.

#  1. Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# # Estilos de grafico
# plt.style.use('seaborn-darkgrid')

# 2. Crearl el cojunto de datos simulados

np.random.seed(42)  # Para reproducibilidad
n = 500  # Número de participantes
#simular puntuaciones de memoria con música y sin música
con_musica = np.random.normal(loc=75, scale=10, size=n).round(2)
sin_musica = np.random.normal(loc=70, scale=10, size=n).round(2)
# Crear un DataFrame
datos = pd.DataFrame({
    'Participante': [f"M{i+1}" for i in range(n)] * 2, # "M1", "M2", ..., "M15"
    "Grupo": ['Con música'] * n + ['Sin música'] * n,
    'Puntuación': np.concatenate([con_musica, sin_musica])
})

# Mostrar las primeras filas del DataFrame
print(datos.head())
datos.to_csv('input/datos_musica_memoria.csv', index=False)

#Paso1.: Indentificar el problem
#Escuchar musica  mejora el rendimiento en memoria a corto plazo?
#Paso2.: Formular la hipotesis 
# H1 = la musis tiene efecto positivo en la memoria a corto plazo
# H0 = la musica no tiene efecto en la memoria a corto plazo

# Paso 3: Define las variables
# Variable independiente: Grupo (Con música, Sin música)
# Variable dependiente: Puntuación de memoria

# Paso 4: Disenar el experimento
# Experimento: Asignar aleatoriamente a los participantes a dos grupos (con música y sin música) y medir sus puntuaciones de memoria.


# Paso 5: Recolectar los datos
# Datos ya generados en el paso 2       
# Paso 6: Analizar los datos

resumen = datos.groupby('Grupo')['Puntuación'].describe()
print(resumen)

# Paso 7: Visualizar los datos
plt.figure(figsize=(10, 6))
datos.boxplot(column='Puntuación', by='Grupo', grid=False)
plt.title('Puntuaciones de memoria por grupo')
plt.suptitle('')
plt.xlabel('Grupo')
plt.ylabel('Puntuación de memoria')
plt.tight_layout()
plt.savefig('output/boxplot_musica_memoria.png')
plt.show()

#interpretar los resultados
alpha = 0.05  # Nivel de significancia
grupo_con_musica = datos[datos['Grupo'] == 'Con música']['Puntuación']
grupo_sin_musica = datos[datos['Grupo'] == 'Sin música']['Puntuación']
pval = stats.ttest_ind(grupo_con_musica, grupo_sin_musica, equal_var=False)
print(f"p-valor: {pval.pvalue:.4f}")
if pval.pvalue < alpha:
    print("Rechazamos la hipótesis nula: La música tiene un efecto significativo en las puntuaciones de memoria.")
else:
    print("No rechazamos la hipótesis nula: No hay evidencia suficiente para afirmar que la música afecta las puntuaciones de memoria.")







