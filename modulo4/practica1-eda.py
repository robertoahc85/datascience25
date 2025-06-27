import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Paso 1. Cargo de datos
df=pd.read_csv("entrada/empleados_150.csv")

#Paso 2 .Inspeccion  Inicial
print("---------Primera filas del dataframe")
print(df.head(10))
print(df.describe())
print("+++++++++Informacion del dataframe")
print(df.info())

#Paso3 . Indentificacion de problemas 
print("Valor faltantes por columna")
print(df.isnull().sum())
print("Filas duplicadas",df.duplicated().sum())
print("Tipo de datos por columna")
print(df.dtypes)

#Paso4  Limpieza de los datos
df = df.drop_duplicates() #eliminar duplicados
df['Edad'] = pd.to_numeric(df['Edad'] , errors='coerce')
df['Edad'].fillna(df['Edad'].median(), inplace=True)
df['Salario'].fillna(df['Salario'].median(), inplace=True)

#Paso5 Estadistica descriptiva
print("\n Medidas de tendecia central (Edad):")
print("Media", df["Edad"].mean())
print("Mediana", df["Edad"].median())
print("Moda", df["Edad"].mode())

print("\n Medidas de dispersion (Salario)")
print("Rango", df["Salario"].max)
print("Media", df["Salario"].median())
print("Desviacion estandard:", df["Salario"].std())

#Paso6 Visualizacion
#5.1 Histogramas de edades
sns.histplot(df['Edad'],kde=True)
plt.title("Distribuccion de edad")
plt.xlabel("Edad")
plt.ylabel("Frencuencia")
plt.show()


#5.2 Boxplot de Salario
sns.boxplot(x=df['Salario'])
plt.title("Boxplot de Salario")
plt.xlabel("Salario")
plt.show()

#5.3 Corelacion entre variables
#Scatterplot de edad vs Salario
sns.scatterplot(x='Edad', y='Salario', data=df)
plt.title("Relacion entre Edad y Salario")
plt.xlabel("Edad")
plt.ylabel("Salario")
plt.show()

#5.4 pairplot(Analis Multivariado visual)
sns.pairplot(df[['Edad','Salario']])
plt.suptitle("Distribucion y relacion entre Edad y Salario", y=1.02)
plt.show()