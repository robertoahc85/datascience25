import pandas as pd
datos = sorted([10,12,12,14,15,16,18,17,10,19,100])
df=pd.DataFrame(datos, columns=['Valor'])

Q1= df['Valor'].quantile(0.25)
Q3= df['Valor'].quantile(0.75)
IQR = Q3 - Q1

limite_inferior = Q1 - 1.5 * IQR
limite_superior = Q3 + 1.5 * IQR

outliers = df[(df["Valor"] < limite_inferior) | (df["Valor"] > limite_superior)]
valores_normales = df[(df['Valor'] >= limite_inferior) & (df['Valor'] <= limite_superior)]

#resultado
print("Datos Ordenados")
print(valores_normales)
print(outliers)


