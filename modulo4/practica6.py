# Este programa tiene como objetivo analizar y visualizar la producción de petróleo
# y gas natural en pozos ubicados en diferentes estados de México,
# utilizando técnicas avanzadas de visualización con pandas y matplotlib.

# A partir de un archivo CSV simulado (produccion_petroleo_mexico.csv) con 500 registros sintéticos, 
# que incluye variables clave como:

# Producción diaria de petróleo (Oil_Production) # barriles

# Producción diaria de gas (Gas_Production) #pie cubico normales

# Corte de agua (Water_Cut)# fraccion

# Presión en cabeza del pozo (Wellhead_Pressure) # psi

# Temperatura del yacimiento (Reservoir_Temperature)# grados Centigrados

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#Cargar el conjunto de datos
df = pd.read_csv('entrada/produccion_petroleo_mexico.csv')
# Filtra  un pozo específico de linea y area
pozo_norte = df[df['Well_ID'] == 'Pozo Norte'].sort_values("Day")

# Grafico de Dispersion de Producción de Producción vs Presion en cabeza del pozo
plt.figure(figsize=(8, 5))
plt.scatter(pozo_norte['Wellhead_Pressure'], pozo_norte['Oil_Production'],cmap='viridis', alpha=0.7)
plt.xlabel('Presión en cabeza del pozo (psi)')
plt.ylabel('Producción de Petróleo (barriles)')
plt.title('Producción de Petróleo vs Presión en Cabeza del Pozo')
plt.grid()
plt.savefig('salida/produccion_vs_presion.png', dpi=300, bbox_inches='tight')
plt.show()
# Crea lista de pozos unicos
pozos_unicos = df['Well_ID'].unique()

# Gráfico de líneas de producción de petróleo  en Pozo Norte
plt.figure(figsize=(8, 5))
plt.plot(pozo_norte['Day'], pozo_norte['Oil_Production'], label='Producción de diaria', color='blue')
plt.xlabel('Día')
plt.ylabel('Producción de Petróleo (barriles)')
plt.title('Producción Diaria de Petróleo en Pozo Norte')
plt.xlim(0,100) # Ajusta los
plt.legend()
plt.grid()
plt.savefig('salida/produccion_diaria_pozo_norte.png', dpi=300, bbox_inches='tight')
plt.show()


#Grafico de superficie 3D de producción de petróleo 
fig =plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X3,Y3 = np.meshgrid(np.linspace(1,100,50), np.linspace(1500,3500,50))
Z3 = 1000 * np.sin(X3/15) * 200 - np.cos(Y3 -2400)
ax.plot_surface(X3, Y3, Z3 , cmap='viridis', edgecolor='none')
ax.set_xlabel('Día')
ax.set_ylabel('Presión en cabeza del pozo (psi)')
ax.set_zlabel('Producción de petróleo (barriles)')
ax.set_title('Superficie 3D de Producción de Petróleo')
plt.savefig('salida/superficie_3D_produccion_petroleo.png', dpi=300, bbox_inches='tight')
plt.show()


# Gráfico de dispersión 3D de producción de petróleo y gas
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pozo_norte['Day'], pozo_norte['Wellhead_Pressure'], pozo_norte['Oil_Production'], c='blue', marker='o', label='Pozo Norte')
ax.set_xlabel('Día')
ax.set_ylabel('Presión en cabeza del pozo (psi)')
ax.set_zlabel('Producción de petróleo (barriles)')
ax.set_title('Dispersión 3D de Producción de Petróleo y Gas')
ax.legend()
plt.savefig('salida/dispersión_3D_produccion_petroleo_gas.png', dpi=300, bbox_inches='tight')
plt.show()