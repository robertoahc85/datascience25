import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
 
#Analice la distribución de 200 muestras de 50 personas  con un 30% de fumadores.

np.random.seed(42)
n= 50
p= 0.30

props = [np.mean(np.random.binomial(1,p,n)) for _ in range(200)]
props_df = pd.DataFrame({'proporcion': props})

#Analisis
print(f"Media de proporciones: {props_df['proporcion'].mean():.3f}")
print(f"Desviación estándar de proporciones: {props_df['proporcion'].std():.3f}")

sns.histplot(props_df['proporcion'], kde=True, color = 'salmon', edgecolor='black')

plt.pyplot.xlabel('Proporción de fumadores')
plt.pyplot.title('Distribución de proporciones de fumadores en 200 muestras')
plt.pyplot.ylabel('Frencencia')
plt.pyplot.savefig('graficas/distrubu.png')
plt.pyplot.show()