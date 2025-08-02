# Métodos de Validación:
# - Train/Test Split (Hold-out)
# - K-Fold Cross-Validation (Stratified)
# - Leave-One-Out Cross-Validation (LOO)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, train_test_split, cross_val_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    make_scorer
)

# 1. Crear datos simulados
X, y = make_classification(
    n_samples=50, n_features=5, n_informative=4, n_redundant=0,
    random_state=42, flip_y=0.02, class_sep=1.0
)

# Modelo base
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# 2. Diccionario para almacenar resultados
metricas = ["Accuracy", "Precision", "Recall", "F1-score"]
resultados = {m: [] for m in metricas}
nombres = []

# 3. Método Hold-out (Train/Test Split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Clases reales:", np.unique(y_test))
print("Clases predichas:", np.unique(y_pred))

resultados["Accuracy"].append(accuracy_score(y_test, y_pred))
resultados["Precision"].append(precision_score(y_test, y_pred))
resultados["Recall"].append(recall_score(y_test, y_pred))
resultados["F1-score"].append(f1_score(y_test, y_pred))
nombres.append("Hold-out")

# 4. Método: Stratified K-Fold Cross-Validation (k=5)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

f1s = cross_val_score(model, X, y, cv=skf, scoring="f1")
accs = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
precs = cross_val_score(model, X, y, cv=skf, scoring="precision")
recs = cross_val_score(model, X, y, cv=skf, scoring="recall")

resultados["Accuracy"].append(accs.mean())
resultados["Precision"].append(precs.mean())
resultados["Recall"].append(recs.mean())
resultados["F1-score"].append(f1s.mean())
nombres.append("Stratified K-Fold")

# 5. Método: Leave-One-Out (LOO)
loo = LeaveOneOut()

f1s = cross_val_score(model, X, y, cv=loo, scoring=make_scorer(f1_score, zero_division=0))
accs = cross_val_score(model, X, y, cv=loo, scoring="accuracy")
precs = cross_val_score(model, X, y, cv=loo, scoring=make_scorer(precision_score, zero_division=0))
recs = cross_val_score(model, X, y, cv=loo, scoring=make_scorer(recall_score, zero_division=0))

resultados["Accuracy"].append(accs.mean())
resultados["Precision"].append(precs.mean())  # Corrección: "Precission" → "Precision"
resultados["Recall"].append(recs.mean())
resultados["F1-score"].append(f1s.mean())
nombres.append("Leave-One-Out")

# 6. Crear DataFrame de resultados
df_resultados = pd.DataFrame(resultados, index=nombres).round(3)
tabla_html = df_resultados.to_html(classes="table table-bordered", border=0)

# 7. Gráfica comparativa
plt.figure(figsize=(10, 6))
df_melt = df_resultados.reset_index().melt(id_vars="index", var_name="Métrica", value_name="Valor")
sns.barplot(data=df_melt, x="index", y="Valor", hue="Métrica")
plt.xlabel("Método de Validación")
plt.ylabel("Valor promedio de la métrica")
plt.title("Comparación de Métodos de Validación Cruzada")
plt.legend(title="Métricas")
plt.tight_layout()
plt.savefig("grafica_validacion.png")
plt.close()

# 8. Interpretación dinámica
interpretaciones = ""
for i, row in df_resultados.iterrows():
    interp = f"""
    <div class="alert alert-info">
    <h5>{i}</h5>
    <ul>
        <li>Accuracy Promedio: {row['Accuracy']:.3f}</li>
        <li>Precision Promedio: {row['Precision']:.3f}</li>
        <li>Recall Promedio: {row['Recall']:.3f}</li>
        <li>F1-score Promedio: {row['F1-score']:.3f}</li>
    </ul>
    </div>
    """
    interpretaciones += interp
    

# 9. Dashboard en HTML
with open("dashboard_validacion_cruzada.html", "w") as f:
    f.write(f"""
    <html>
    <head>
        <title>Dashboard: Métodos de Validación Cruzada</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.7/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-5">
            <h1>Dashboard de Validación Cruzada</h1>
            <p>Comparación de 4 métricas para distintos métodos</p>
            <h2>Tabla de Resultados</h2>
            {tabla_html}
            <hr>
            <h2>Gráfica Comparativa</h2>
            <img src="grafica_validacion.png" class="img-fluid">
            <hr>
            <h2>Interpretación Dinámica</h2>
            {interpretaciones}
        </div>
    </body>
    </html>
    """)
