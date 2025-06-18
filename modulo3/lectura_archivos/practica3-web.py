import pandas as pd
# cargar desde web
url = "https://es.wikipedia.org/wiki/Anexo:Pa%C3%ADses_y_territorios_dependientes_por_poblaci%C3%B3n" 
tablas = pd.read_html(url)
# imprimir el número de tablas
print(f"Número de tablas encontradas: {len(tablas)}")
# imprimir la primera tabla
df =tablas[0]
print(df.head())
# guardar la tabla en un archivo excel
df.to_excel('salidas/poblacion.xlsx', index=False)
df.to_csv('salidas/poblacion.csv', index=False)  # guardar la tabla en un archivo csv
# guardar la tabla en un archivo csv



