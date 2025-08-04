# ============================================
# Reto: Codificación de Variables Categóricas
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 1. Crear un dataset simulado con variables categóricas y numéricas
np.random.seed(42)
df = pd.DataFrame({
    "cliente_id": range(1, 11),
    "genero": np.random.choice(["masculino", "femenino"], size=10),
    "plan": np.random.choice(["básico", "estándar", "premium"], size=10),
    "estado": np.random.choice(["activo", "cancelado", "pendiente"], size=10),
    "score_crediticio": np.random.randint(300, 850, size=10)
})

print("📋 Dataset original:")
print(df)

# ===============================================
# TEORÍA: ¿Qué es Label Encoding?
"""
Label Encoding convierte cada clase (texto) en un número entero.
Ejemplo: ["masculino", "femenino"] → [1, 0]
Ventajas:
- Simple y útil para variables ordinales (bajo < medio < alto)
Desventajas:
- El modelo podría malinterpretar que hay orden en los valores
"""

# 2. Aplicar Label Encoding a "genero" y "estado"
le_genero = LabelEncoder()
le_estado = LabelEncoder()

# Transformar las columnas y guardar en nuevas columnas codificadas
df["genero_encoded"] = le_genero.fit_transform(df["genero"])  # masculino=1, femenino=0
df["estado_encoded"] = le_estado.fit_transform(df["estado"])  # orden arbitrario

print("\n🔤 Label Encoding aplicado a 'genero' y 'estado':")
print(df[["genero", "genero_encoded", "estado", "estado_encoded"]])

# ===============================================
# TEORÍA: ¿Qué es One-Hot Encoding?
"""
One-Hot Encoding convierte una columna categórica en columnas binarias.
Ejemplo:
    plan = ["básico", "premium"] → plan_básico=1, plan_estándar=0, plan_premium=0
Ventajas:
- No impone orden artificial
- Útil para modelos lineales y redes neuronales
Desventajas:
- Aumenta dimensionalidad cuando hay muchas categorías
"""

# 3. Aplicar One-Hot Encoding a las columnas "plan" y "estado"
columnas = ["plan", "estado"]
onehot = OneHotEncoder(sparse_output=False, drop=None)  # Codificación completa, sin eliminar ninguna clase

# Crear transformador que aplicará OneHotEncoder a columnas seleccionadas
transformador = ColumnTransformer(
    transformers=[
        ("onehot", onehot, columnas)
    ],
    remainder="passthrough"  # Mantener el resto de columnas sin cambio
)

# Aplicar la transformación al dataset
datos_transformados = transformador.fit_transform(df[columnas + ["score_crediticio"]])

# Obtener nombres de las columnas codificadas
columnas_onehot = onehot.get_feature_names_out(columnas)

# Crear nuevo DataFrame con los datos codificados
df_onehot = pd.DataFrame(datos_transformados, columns=list(columnas_onehot) + ["score_crediticio"])

print("\n🧊 One-Hot Encoding aplicado a 'plan' y 'estado':")
print(df_onehot)

# ===============================================
# TEORÍA: ¿Qué son las variables dummy?
"""
Las variables dummy son una forma práctica de aplicar One-Hot Encoding
usando pandas. Permite eliminar una categoría por variable si se desea
evitar multicolinealidad (drop_first=True).
"""

# 4. Crear variables dummy usando pandas.get_dummies
df_dummies = pd.get_dummies(df[["genero", "plan", "estado"]], drop_first=True)

print("\n🐼 Variables Dummy con pandas.get_dummies:")
print(df_dummies)

# ===============================================
# Comparación final de formas
# ===============================================
print("\n📐 Comparación de formas de codificación:")
print("Original:", df.shape)
print("Label Encoding:", df[["genero_encoded", "estado_encoded"]].shape)
print("One-Hot Encoding:", df_onehot.shape)
print("Dummy Variables:", df_dummies.shape)
