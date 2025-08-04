# ============================================================
# Procesamiento y Regresión de Cosechas con Dashboard Visual
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, Normalizer,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Crear carpetas necesarias
os.makedirs("input", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("dashboard", exist_ok=True)

# 1. Leer archivo
df = pd.read_csv("input/Cosechas_Cosechas_2023b.csv", skiprows=1)
df.columns = [
    "Index", "Codigo_Centro", "Empresa", "Especie",
    "Toneladas_Cosechadas", "Mes_Inicio", "Mes_Fin", "Periodo"
]

# 2. Procesar columna numérica
df["Toneladas_Cosechadas"] = pd.to_numeric(df["Toneladas_Cosechadas"], errors="coerce")
df["Toneladas_Cosechadas"].fillna(df["Toneladas_Cosechadas"].median(), inplace=True)
umbral = df["Toneladas_Cosechadas"].quantile(0.99)
df["Toneladas_Cosechadas"] = np.where(
    df["Toneladas_Cosechadas"] > umbral,
    df["Toneladas_Cosechadas"].median(),
    df["Toneladas_Cosechadas"]
)

# 3. Escalamiento y transformaciones
df["Cosecha_StandardScaler"] = StandardScaler().fit_transform(df[["Toneladas_Cosechadas"]])
df["Cosecha_MinMaxScaler"] = MinMaxScaler().fit_transform(df[["Toneladas_Cosechadas"]])
df["Cosecha_RobustScaler"] = RobustScaler().fit_transform(df[["Toneladas_Cosechadas"]])
df["Cosecha_log"] = np.log1p(df["Toneladas_Cosechadas"])
df["Cosecha_sqrt"] = np.sqrt(df["Toneladas_Cosechadas"])
df["Cosecha_normalized"] = Normalizer().fit_transform(df[["Toneladas_Cosechadas"]])

# 4. Gráfica KDE para comparar técnicas de escalamiento
plt.figure(figsize=(10, 5))
sns.kdeplot(df["Toneladas_Cosechadas"], label="Original", linewidth=2)
sns.kdeplot(df["Cosecha_StandardScaler"], label="StandardScaler")
sns.kdeplot(df["Cosecha_MinMaxScaler"], label="MinMaxScaler")
sns.kdeplot(df["Cosecha_RobustScaler"], label="RobustScaler")
plt.legend()
plt.title("Comparación de Técnicas de Escalamiento")
plt.tight_layout()
plt.savefig("dashboard/cosechas_escalado.png")
plt.close()

# 5. Regresión lineal con One-Hot Encoding
df_modelo = df[["Empresa", "Especie", "Periodo", "Toneladas_Cosechadas"]].dropna()
df_encoded = pd.get_dummies(df_modelo, columns=["Empresa", "Especie", "Periodo"], drop_first=True)

X = df_encoded.drop("Toneladas_Cosechadas", axis=1)
y = df_encoded["Toneladas_Cosechadas"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# 6. Interpretación dinámica
resumen = df[[
    "Toneladas_Cosechadas", "Cosecha_StandardScaler", "Cosecha_MinMaxScaler",
    "Cosecha_RobustScaler", "Cosecha_log", "Cosecha_sqrt", "Cosecha_normalized"
]].describe().T.round(3)

resumen["Técnica"] = resumen.index
resumen.rename(columns={"mean": "Media", "std": "Desviación estándar"}, inplace=True)

interpretacion_html = "<div class='card mt-4'><div class='card-body'><h5 class='card-title'>📌 Interpretación Automática</h5><ul>"
for _, row in resumen.iterrows():
    tecnica = row["Técnica"]
    media = row["Media"]
    std = row["Desviación estándar"]
    if tecnica == "Toneladas_Cosechadas":
        msg = f"<strong>{tecnica}:</strong> media={media:.1f}, STD={std:.1f}. Requiere escalamiento."
    elif "StandardScaler" in tecnica:
        msg = f"<strong>{tecnica}:</strong> centrado en 0, desviación ≈1. Ideal para regresión."
    elif "MinMaxScaler" in tecnica:
        msg = f"<strong>{tecnica}:</strong> escala [0-1]. Útil para redes neuronales."
    elif "RobustScaler" in tecnica:
        msg = f"<strong>{tecnica}:</strong> robusto a outliers. Desviación ≈ {std:.2f}."
    elif "log" in tecnica:
        msg = f"<strong>{tecnica}:</strong> reduce sesgo positivo y valores grandes."
    elif "sqrt" in tecnica:
        msg = f"<strong>{tecnica}:</strong> suaviza la distribución."
    elif "normalized" in tecnica:
        msg = f"<strong>{tecnica}:</strong> vector normalizado. Útil para distancias."
    interpretacion_html += f"<li>{msg}</li>"
interpretacion_html += "</ul></div></div>"

# 7. Tablas HTML
tabla_resumen_html = resumen[["Media", "Desviación estándar"]].round(3).to_html(classes="table table-bordered table-sm")
tabla_muestra_html = df[["Empresa", "Especie", "Toneladas_Cosechadas"]].head(10).to_html(classes="table table-striped")

# 8. Generación de Dashboard HTML
with open("dashboard/dashboard_cosechas.html", "w", encoding="utf-8") as f:
    f.write(f"""
    <!DOCTYPE html>
    <html><head><meta charset="utf-8">
    <title>Dashboard Cosechas 2023</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    </head><body><div class="container mt-4">
    <h1>🌾 Dashboard: Cosechas 2023</h1>

    <h2>📋 Muestra del Dataset</h2>
    {tabla_muestra_html}

    <hr><h2>📈 Comparación de Escalamiento</h2>
    <img src="cosechas_escalado.png" class="img-fluid"/>

    <hr><h2>📊 Estadísticas por Técnica</h2>
    {tabla_resumen_html}

    {interpretacion_html}

    <hr><h2>🔢 Regresión Lineal</h2>
    <div class="alert alert-info">
        <strong>R² Score:</strong> {r2:.3f}<br>
        <strong>Error Cuadrático Medio (MSE):</strong> {mse:.2f}
    </div>
    <img src="regresion_resultado.png" class="img-fluid mt-3"/>

    <hr><h2>✅ Conclusión</h2>
    <div class="alert alert-success">
        Se aplicó un flujo de procesamiento y predicción sobre la variable <strong>Toneladas Cosechadas</strong>.
        El modelo de regresión ofrece una aproximación inicial al comportamiento de las cosechas.
    </div>
    </div></body></html>
    """)

# 9. Guardar dataset procesado
df.to_csv("output/cosechas_procesadas.csv", index=False)

print("✅ Dashboard con regresión generado en: dashboard/dashboard_cosechas.html")
print("✅ Dataset procesado guardado en: output/cosechas_procesadas.csv")
