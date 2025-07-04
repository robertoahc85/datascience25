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
# # Gráfico 1: Histplot + kde  - Distribución del porcentaje a favor
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

#grafico 3: Regplot - Relación entre participación y porcentaje a favor
plt.figure(figsize=(9, 5))
sns.regplot(x='turnout',
            y='percent_favor',
            data=df,
            scatter_kws={'alpha':0.5, 'color':'darkorange'}, #Configuración de puntos transparencia y color
            line_kws={'color':'red', 'linewidth':2}, #Configuración de la línea de regresión
        )
plt.title("Relación entre Participación Electoral y Porcentaje a Favor", fontsize=16,pad=15)
plt.xlabel("Participación Electoral (%)", fontsize=14)
plt.ylabel("Porcentaje a Favor (%)", fontsize=14)
plt.tight_layout()
plt.savefig("graficas/regplot_turnout_percent_favor.png")
plt.show()

# Gráfico 4: Barplot - Promedio de porcentaje a favor por región
#Calculamos el promedio de porcentaje a favor por región
region_avg = (df.groupby('region',as_index=False)).mean(numeric_only=True).sort_values(by='percent_favor', ascending=False)
plt.figure(figsize=(11, 6))
sns.barplot(x='percent_favor', 
            y='region', 
            data=region_avg, 
            palette='viridis', #Paleta de colores
            edgecolor='black', #Color del borde de las barras
            )
plt.title("Promedio de Porcentaje a Favor por Región", fontsize=16,pad=15)
plt.xlabel("Porcentaje a Favor (%)", fontsize=14)
plt.ylabel("Región", fontsize=14)
plt.tight_layout()
plt.savefig("graficas/barplot_region_percent_favor.png")
plt.show()

# Gráfico 5: Violinplot - Distribución del porcentaje a favor por región
plt.figure(figsize=(12, 6))
sns.violinplot(x='region', 
               y='percent_favor',
               data=df,
               cut=0, #Corta los valores extremos
               palette='pastel', #Paleta de colores
               inner='quartile', #Muestra los cuartiles internos
)
plt.title("Distribución del Porcentaje a Favor por Región", fontsize=16,pad=15)
plt.xlabel("Región", fontsize=14)
plt.ylabel("Porcentaje a Favor (%)", fontsize=14)
plt.xticks(rotation=45) #Rota las etiquetas del eje x
plt.tight_layout()
plt.savefig("graficas/violinplot_region_percent_favor.png")
plt.show()

# Gráfico 6: Triada clave (% a favor, %Contra, Participación)

sns.pairplot(df,
             vars=['percent_favor', 'percent_against', 'turnout'],
             hue='region', #Colorea por región,
             diag_kind='kde', #Curva KDE en la diagonal,
             height=2.4,
             corner=True, #Muestra solo la mitad inferior
)
plt.suptitle("Relaciones bivariada clave por Region", fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig("graficas/pairplot_triada_clave.png")
plt.show()

#grafico 7: FacetGrid - Densidad a favor  segun rango de participación
#Creamos 4 densidades de participación (bins)
df['turnout_bins'] = pd.cut(df['turnout'],
                            bins=[60, 70, 80, 90, 100],#   Rangos de participación
                            labels=['60-70%', '70-80%', '80-90%', '90-100%'], #Etiquetas de los rangos
)
g = sns.FacetGrid(df, col='turnout_bins', col_wrap=2, height=3.2, sharex=True)# Fuera que  todo lo subgrafico tenga el mismo eje x

g.map_dataframe(
    sns.kdeplot,
    x='percent_favor',
    fill=True,
    clip=(30,70)
    ) #Curva KDE

g.set_titles(col_template="{col_name} participacion") #Título de cada subgráfico
g.fig.subplots_adjust(top=0.9) #Ajusta el espacio superior
g.fig.suptitle("Distribución del Porcentaje a Favor por Rango de Participación", fontsize=16)
g.savefig("graficas/facetgrid_turnout_bins_percent_favor.png")
plt.show()
