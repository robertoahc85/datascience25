import pandas as pd
# data = {"Nombre": ["Juan", "Ana", "Pedro"],
#         "Curso": ["Matematicas", "Historia", "Ciencias"],
#         "Calificacion": [90, 70, 70]}
df_entrada = pd.read_csv("entradas/estudiantes_100.csv")
df = pd.DataFrame(df_entrada)
# print(df)
print(df.describe())
df.to_csv("salidas/estudiantes.csv", index=False)
df_filtrado = df[df["Calificacion"] > 80]
cal_min=df_filtrado['Calificacion'].min() 
cal_max =df_filtrado['Calificacion'].max() 
nuevo_min = 0
nuevo_max = 10
# df['Calif_0_10'] = (df['Calificacion'] - cal_min) / (cal_max - cal_min) * (nuevo_max - nuevo_min) + nuevo_min

df_ordenado = df_filtrado.sort_values(by="Calificacion", ascending=False)
print(df_ordenado)
df_ordenado.to_csv("salidas/estudiantes_filtrados.csv", index=False)
promedio_cursos = df["Calificacion"].mean()
print(f"El promedio de las calificaciones es: {promedio_cursos:.2f}")
promedio_cursos_por_curso = df.groupby("Curso")["Calificacion"].mean()
print("Promedio de calificaciones por curso:")
print(promedio_cursos_por_curso)

#Valor nulos
print("Valores nulos en cada columna:")                 
print(df.isnull())
print(df.isnull().sum())
df_sin_nulos = df.dropna()
df_sin_nulos.to_csv("salidas/estudiantes_sin_nulos.csv", index=False)

df_sin_columnas_nulas = df.dropna(axis=1)
df_sin_columnas_nulas.to_csv("salidas/estudiantes_sin_columnas_nulas.csv", index=False)


df['Calificacion'] = df['Calificacion'].fillna(df['Calificacion'].mean())
df['Nombre'] = df['Nombre'].fillna("Desconocido")
df['Curso'] = df['Curso'].fillna("Sin curso Asignado")
df.to_csv("salidas/estudiantes_relleno.csv", index=False)
promedio_por_nombre = df.groupby('Nombre')['Calificacion'].mean().reset_index()
promedio_por_nombre = promedio_por_nombre.sort_values(by='Calificacion',ascending=False)
print(promedio_por_nombre)

