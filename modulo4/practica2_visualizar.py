# Paso 1. importa libreria necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Paso2: Leer el dataset con variables mixta
df= pd.read_csv('entrada/dataset_con_outliers.csv')

#Paso3 : Frencuencia de variables categoricas
freq_color = df['Color'].value_counts() #Nominales
freq_satifacion = df['Satisfaction'].value_counts(sort=False) #Ordinales

print(freq_color)
print(freq_satifacion)

#Paso4 : Funcion para obtener stadisticas descriptivas
def calcular_estadisticas(dataframe):
    return pd.DataFrame({
        'mean': dataframe.mean(),
        'median': dataframe.median(),
        'moda': dataframe.mode().iloc[0],
        'variance': dataframe.var(ddof=1),#n-1 muestra
        'std_dev': dataframe.std(ddof=1),
        'IQR': dataframe.quantile(0.75) - dataframe.quantile(0.25)
    })

#Estadistica con outlier
estadisticas_originales = calcular_estadisticas(df[['Children', 'Temperature' ]])
print(estadisticas_originales)

#Paso5.Detectar outliera en columna numerica
def detectar_outlier_iqr(serie):
    Q1 = serie.quantile(0.25)
    Q3 = serie.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = serie[(serie < limite_inferior) | (serie > limite_superior)]
    return outliers, limite_inferior,limite_superior


out_children, li_c ,ls_c = detectar_outlier_iqr(df['Children'])
out_temp, li_t ,ls_t = detectar_outlier_iqr(df['Temperature'])


print(" Children:\n",out_children.values)
print(" Temperature:\n",out_temp.values)

#Eliminar registro  con outlier
df_clean = df[
    (df['Children'].between(li_c,ls_c)) &
    (df['Temperature'].between(li_t,ls_t)) ]

estadisticas_limpia = calcular_estadisticas(df_clean[['Children', 'Temperature']])
print(estadisticas_limpia)


# Paso 6: Crear Pdf con analisis 
with PdfPages('salida/analisis_reporte.pdf') as pdf:
    #Pagina 1: Portada
    fig = plt.figure(figsize=(11,8.5))
    plt.suptitle("Reporte Analisis de Datos", fontsize=16, y=0.95)
    plt.text(0.1, 0.6, f"Dataset Original:{len(df)} registros", fontsize=12)
    plt.text(0.1, 0.4, f"Dataset limpios:{len(df_clean)} registros", fontsize=12)
    plt.axis('off')
    pdf.savefig(fig)
    plt.close()
    
    
#Pagina 2    
    



