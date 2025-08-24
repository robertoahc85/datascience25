# ==============================================
# 1) Importar librerías
# ==============================================
import numpy as np                     # Para generar datos numéricos y arreglos
import matplotlib.pyplot as plt        # Para visualización de gráficos

# Librerías de scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV  # División de datos y búsqueda de hiperparámetros
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Métricas de evaluación
from sklearn.ensemble import GradientBoostingRegressor  # Modelo de Gradient Boosting para regresión
from sklearn.preprocessing import StandardScaler        # Escalador de características
from sklearn.pipeline import Pipeline                   # Para encadenar pasos (escalado + modelo)

# Fijamos semilla para reproducibilidad (los resultados se repiten igual en cada ejecución)
np.random.seed(42)


# ==============================================
# 2) Generar datos sintéticos (regresión)
#    Relación no lineal con ruido
# ==============================================
n = 600
# Creamos una variable independiente X de -3 a 3 (600 valores)
X = np.linspace(-3, 3, n).reshape(-1, 1)

# Variable dependiente y = función seno con algo de ruido gaussiano
y = np.sin(1.5 * X).ravel() + 0.3 * np.random.randn(n)


# ==============================================
# 3) Dividir en entrenamiento y prueba
# ==============================================
# train_test_split divide el dataset en entrenamiento (75%) y prueba (25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ==============================================
# 4) Crear y entrenar el modelo de Gradient Boosting
# ==============================================
# Usamos un Pipeline: 1) escalado, 2) modelo
# (el escalado no es obligatorio, pero es buena práctica si después queremos probar otros modelos)
gbr = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),  # Normaliza X
    ("model", GradientBoostingRegressor(
        random_state=42,   # semilla
        n_estimators=200,  # número de árboles a usar
        learning_rate=0.08, # cuánto contribuye cada árbol al resultado final
        max_depth=3,       # profundidad de los árboles individuales (complejidad)
        subsample=0.9      # fracción de muestras para entrenar cada árbol (reduce sobreajuste)
    ))
])

# Entrenamos el modelo con los datos de entrenamiento
gbr.fit(X_train, y_train)


# ==============================================
# 5) Evaluar el modelo
# ==============================================
# Hacemos predicciones en entrenamiento y prueba
pred_train = gbr.predict(X_train)
pred_test  = gbr.predict(X_test)

# Calculamos métricas en el conjunto de prueba
mae  = mean_absolute_error(y_test, pred_test)                    # Error absoluto medio
rmse = np.sqrt(mean_squared_error(y_test, pred_test))           # Raíz del error cuadrático medio
r2   = r2_score(y_test, pred_test)                              # R² (qué tan bien explica el modelo la varianza)

# Imprimimos métricas
print("=== Métricas en TEST ===")
print(f"MAE : {mae:0.4f}")
print(f"RMSE: {rmse:0.4f}")
print(f"R²  : {r2:0.4f}")


# ==============================================
# 6) Visualizar las predicciones
# ==============================================

# a) Visualizar curva real vs predicción
# Creamos una grilla de valores X ordenados para dibujar la curva del modelo
X_grid = np.linspace(X.min(), X.max(), 400).reshape(-1, 1)
y_hat_grid = gbr.predict(X_grid)

plt.figure(figsize=(7, 4.5))
plt.scatter(X_train, y_train, s=12, alpha=0.4, label="Train (real)")  # Datos de entrenamiento reales
plt.scatter(X_test,  y_test,  s=18, alpha=0.8, label="Test (real)")   # Datos de prueba reales
plt.plot(X_grid, y_hat_grid, linewidth=2.5, label="Predicción (GBR)") # Curva aprendida por el modelo
plt.title("Gradient Boosting Regressor: ajuste en 1D")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.show()

# b) Dispersión de y_real vs y_pred
plt.figure(figsize=(5.5, 5))
plt.scatter(y_test, pred_test, s=18, alpha=0.8)  # puntos predicción vs real
# Dibujamos la línea ideal y = x
min_v = min(y_test.min(), pred_test.min())
max_v = max(y_test.max(), pred_test.max())
plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")
plt.title("y_real vs y_pred (TEST)")
plt.xlabel("y_real")
plt.ylabel("y_pred")
plt.tight_layout()
plt.show()


# ==============================================
# 7) Mejorar el modelo (búsqueda de hiperparámetros)
# ==============================================
# Definimos un espacio de búsqueda para los hiperparámetros
param_grid = {
    "model__n_estimators": [150, 250, 400],   # número de árboles
    "model__learning_rate": [0.03, 0.06, 0.1], # tasa de aprendizaje
    "model__max_depth": [2, 3, 4],            # profundidad de cada árbol
    "model__subsample": [0.8, 1.0]            # proporción de datos por árbol
}

# Definimos un pipeline base (escalado + modelo)
gbr_base = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("model", GradientBoostingRegressor(random_state=42))
])

# GridSearchCV probará todas las combinaciones de hiperparámetros
grid = GridSearchCV(
    estimator=gbr_base,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",  # optimizamos para minimizar MSE
    cv=5,                              # validación cruzada con 5 particiones
    n_jobs=-1                          # usa todos los núcleos disponibles
)

# Entrenamos la búsqueda
grid.fit(X_train, y_train)

# Extraemos el mejor modelo y sus parámetros
best_model = grid.best_estimator_
best_params = grid.best_params_

print("\n=== Mejora del modelo (GridSearchCV) ===")
print("Mejores hiperparámetros:", best_params)

# Re-evaluamos con el mejor modelo encontrado
best_pred_test = best_model.predict(X_test)
best_rmse = np.sqrt(mean_squared_error(y_test, best_pred_test))  # Calculamos RMSE manualmente
best_r2 = r2_score(y_test, best_pred_test)

print(f"RMSE (antes): {rmse:0.4f}  ->  RMSE (mejorado): {best_rmse:0.4f}")
print(f"R²   (antes): {r2:0.4f}    ->  R²   (mejorado): {best_r2:0.4f}")