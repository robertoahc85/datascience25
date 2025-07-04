# Reto: Análisis Visual del Plebiscito Constitucional Chile 2023
# Objetivo general:
# Explorar la relación entre la participación electoral y
# el apoyo a la propuesta constitucional en el plebiscito chileno de 2023,
# utilizando visualizaciones estadísticas generadas con Pandas y Seaborn.

# Archivo de trabajo:
# 📄 plebiscito_chile_2023_400reg.csv
# Contiene 400 registros sintéticos pero basados en datos reales de regiones de Chile. 
# Cada fila representa un resumen de una mesa o sección electoral.
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
import os

# Ajustamos el estilo de los gráficos
sns.set_theme(style="whitegrid", context="talk")
# Cargamos el csv
df =pd.read_csv("entrada/plebiscito_chile_2023_400reg.csv")
print("Primeras Filas del dataset",df.head(5))

# Creamos carpeta donde guardar las graficas
os.makedirs("graficas", exist_ok=True)
# Gráfico 1: Histplot + kde  - Distribución del porcentaje a favor
sns.histplot(df['percent_favor'], 
             kde=True, #Agrega la curva KDE
             color='royalblue', #Color de barra del histogram
             bins=25, #Número de barras del histograma
             edgecolor='white',#Color del borde de las barras
)        
plt.title("Distribución del Porcentaje a Favor del Pelebiscito Chile 2023", fontsize=16,pad=15)
plt.xlabel("Voto a Favor (%)", fontsize=14)
plt.ylabel("Frecuencia", fontsize=14)
plt.tight_layout()
plt.savefig("graficas/histograma_percent_favor.png")
plt.show()

# Gráfico 2: Kde - Distribución del participación electoral
plt.figure(figsize=(9, 5))
sns.kdeplot(df['turnout'],
            shade=True, #Sombreado debajo de la curva
            color='seagreen', #Color de la curva
            
)
plt.title("Distribución de la Participación Electoral en el Plebiscito Chile 2023", fontsize=16,pad=15)
plt.xlabel("Participación Electoral (%)", fontsize=14)
plt.ylabel("Densidad", fontsize=14)
plt.tight_layout()
plt.savefig("graficas/kde_turnout.png")
plt.show()  
