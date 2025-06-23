import pandas as pd
df = pd.read_csv("entradas/ejemplo_datos_outliers.csv")
df["Salario"]= pd.to_numeric(df["Salario"], errors="coerce") 
q1 = df["Salario"].quantile(0.25)
q2 = df["Salario"].quantile(0.50)
q3 = df["Salario"].quantile(0.75)

iqr = q3 -q1
limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

outliers = df[(df["Salario"] < limite_inferior) | (df["Salario"] > limite_superior)]
valores_normales = df[(df['Salario'] >= limite_inferior) & (df['Salario'] <= limite_superior)]


print("Q1:",q1)
print("Q2 (Mediana):",q2)
print("Q3:",q3)
print("IQR:",iqr)
print("Outlier encontrados:")
print(outliers[["ID", "Salario"]])


