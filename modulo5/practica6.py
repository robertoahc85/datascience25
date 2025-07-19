# Una compañía afirma que el tiempo promedio de entrega es 30 minutos. Una muestra
# de 25 entregas muestra un tiempo promedio de 32 minutos con una desviación estándar de
# 5 minutos. Se desea probar esta afirmación con un nivel de significancia del 5 % (α = 0,05)
# usando una prueba t de una muestra.
# Hipótesis Nula (H0): µ = 30 (el tiempo promedio de entrega es 30 minutos). -
# # Hipótesis Alternativa (H1): µ ̸= 30 (el tiempo promedio de entrega no es 30 minutos). -
# - Nivel de significancia: α = 0.05
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Definir los paramentros
n=25  #Tamano d ela muestra
mean = 32 #media muestral
mu_0 = 30 #Media hipotetica
std = 5 # desviacion estandar muestral
alpha = 0.05 #Nivel de significancia
df = n - 1 #Grado de libertad

#calcular la estadistica t
# Fórmula t = (̄x -(s / √n)
t_stat = (mean - mu_0) / (std / np.sqrt(n))
print(f" Estadistica calculada:{t_stat:.2f}")

# Generar valores para distribuccion t
x = np.linspace(-4,4,1000) # RAngo de valores para la grafica
y = stats.t.pdf(x,df) # Densidad de probalidad de la probalidad de la distribuccion t

#crear la figura
plt.figure(figsize=(10,6))
plt.plot(x,y, 'b-', label= "Distribuccion t(df=24)") #Graficar la curva t

#Rellena regiones de rechazo
# Calcular los valores críticos para dos colas
t_critical = stats.t.ppf(1 - alpha/2, df)

# Rellenar región de rechazo a la izquierda
plt.fill_between(x, 0, y, where=(x <= -t_critical), color='red', alpha=0.3, label='Región de rechazo (izquierda)')
# Rellenar región de rechazo a la derecha
plt.fill_between(x, 0, y, where=(x >= t_critical), color='red', alpha=0.3, label='Región de rechazo (derecha)')

# Dibujar la estadística t calculada
plt.axvline(t_stat, color='green', linestyle='--', label=f'Estadística t = {t_stat:.2f}')

plt.title('Distribución t de Student y regiones de rechazo')
plt.xlabel('t')
plt.ylabel('Densidad de probabilidad')
plt.legend()
plt.grid(True)
plt.show()

#imprimir decision
critical_value = stats.t.ppf(1 -alpha/2, df)
print(f"Valor critico: {critical_value:.2f}")
if abs(t_stat) < critical_value:
    print("No se rechaza H: no hay evidencia suficiente para rechazar  la afirmacion")
else:
    print("Se rechaza H: Hay evidencia suficiente para rechazar la afirmacion")