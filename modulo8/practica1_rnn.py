# ============================================================
# DIGIT RECOGNIZER (Kaggle MNIST) — MLP desde cero (versión corregida)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) CARGA DE DATOS
# -----------------------------
df = pd.read_csv('data/train.csv')   # ajusta ruta si hace falta
y = df['label'].values              # (N,)
x = df.drop('label', axis=1)        # DataFrame 784 pixeles

print(f"Dataset: {x.shape[0]} filas, {x.shape[1]} columnas")
print(x.head(2))

# -----------------------------
# 2) ONE-HOT + NORMALIZACIÓN
# -----------------------------
X = x.values.astype(np.float32) / 255.0   # [0,1]
Y = np.eye(10, dtype=np.float32)[y]       # one-hot (N,10)

# -----------------------------
# 3) SPLIT TRAIN / VAL
# -----------------------------
np.random.seed(100)
idx = np.random.permutation(X.shape[0])
split = int(0.9 * len(idx))               # 90% train, 10% val
train_idx, val_idx = idx[:split], idx[split:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]
y_val_labels     = y[val_idx]

# -----------------------------
# 4) ACTIVACIONES Y PÉRDIDA
# -----------------------------
def relu(z):
    return np.maximum(0.0, z)

def relu_derivative(z):
    return (z > 0).astype(np.float32)

def softmax(z):
    z_ = z - z.max(axis=1, keepdims=True)      # estabilidad numérica
    e  = np.exp(z_)
    return e / (e.sum(axis=1, keepdims=True) + 1e-12)

def cross_entropy(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))

def accuracy(y_true_labels, y_pred_probs):
    return np.mean(y_true_labels == np.argmax(y_pred_probs, axis=1))

# -----------------------------
# 5) INICIALIZACIÓN TEÓRICA
# -----------------------------
# CORRECCIÓN: Antes -> W = randn(0,1); b = randn
# Motivo: saturación/varianza inestable con sigmoide; sesgos no necesitan aleatoriedad.

def he_init(n_in, n_out):      # para ReLU en ocultas
    std = np.sqrt(2.0 / n_in)
    return (std * np.random.randn(n_in, n_out)).astype(np.float32)

def xavier_init(n_in, n_out):  # para capa softmax (salida)
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out)).astype(np.float32)

# Arquitectura: 784 -> 128 -> 64 -> 10 (un poco más capaz que 16/16)
h1, h2 = 128, 64
lr = 0.01
batch_size = 128
epochs = 20

W0 = he_init(784, h1)                # CORRECCIÓN: He init (antes: randn(784,16))
b0 = np.zeros((1, h1), dtype=np.float32)      # CORRECCIÓN: ceros (antes: randn)

W1 = he_init(h1, h2)                 # CORRECCIÓN: He init (antes: randn(16,16))
b1 = np.zeros((1, h2), dtype=np.float32)      # CORRECCIÓN: ceros

W2 = xavier_init(h2, 10)             # CORRECCIÓN: Xavier para softmax (antes: randn(16,10))
b2 = np.zeros((1, 10), dtype=np.float32)      # CORRECCIÓN: ceros

# -----------------------------
# 6) FEEDFORWARD
# -----------------------------
def forward(Xb):
    z0 = Xb @ W0 + b0      # (B,h1)
    a0 = relu(z0)

    z1 = a0 @ W1 + b1      # (B,h2)
    a1 = relu(z1)

    z2 = a1 @ W2 + b2      # (B,10)
    P  = softmax(z2)       # CORRECCIÓN: Softmax (antes: sigmoid)

    cache = (Xb, z0, a0, z1, a1, z2, P)
    return P, cache

# -----------------------------
# 7) BACKPROP (Softmax + CE)
# -----------------------------
def backward(Yb, cache):
    Xb, z0, a0, z1, a1, z2, P = cache
    B = Xb.shape[0]

    dZ2 = (P - Yb) / B          # CORRECCIÓN: gradiente salida (antes: -(y-ŷ)*σ')
    dW2 = a1.T @ dZ2
    db2 = dZ2.sum(axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_derivative(z1)
    dW1 = a0.T @ dZ1
    db1 = dZ1.sum(axis=0, keepdims=True)

    dA0 = dZ1 @ W1.T
    dZ0 = dA0 * relu_derivative(z0)
    dW0 = Xb.T @ dZ0
    db0 = dZ0.sum(axis=0, keepdims=True)

    return dW0, db0, dW1, db1, dW2, db2

# -----------------------------
# 8) ENTRENAMIENTO (mini-batch)
# -----------------------------
hist_train, hist_val = [], []

for ep in range(1, epochs + 1):
    perm = np.random.permutation(X_train.shape[0])
    Xb_all, Yb_all = X_train[perm], Y_train[perm]

    # mini-batches
    for s in range(0, Xb_all.shape[0], batch_size):
        e = s + batch_size
        Xb = Xb_all[s:e]
        Yb = Yb_all[s:e]

        P, cache = forward(Xb)
        dW0, db0_, dW1, db1_, dW2, db2_ = backward(Yb, cache)

        # update
        W0 -= lr * dW0; b0 -= lr * db0_
        W1 -= lr * dW1; b1 -= lr * db1_
        W2 -= lr * dW2; b2 -= lr * db2_

    # métricas por época
    P_tr, _ = forward(X_train)
    P_va, _ = forward(X_val)

    loss_tr = cross_entropy(Y_train, P_tr)
    loss_va = cross_entropy(Y_val,   P_va)
    acc_tr  = accuracy(y[train_idx], P_tr)
    acc_va  = accuracy(y_val_labels, P_va)

    hist_train.append(loss_tr)
    hist_val.append(loss_va)

    print(f"Época {ep:02d} | loss_tr={loss_tr:.4f} acc_tr={acc_tr*100:5.2f}% | "
          f"loss_val={loss_va:.4f} acc_val={acc_va*100:5.2f}%")

# -----------------------------
# 9) VISUALIZACIÓN EDUCATIVA (igual que tenías)
# -----------------------------
i = 0
img = X[i].reshape(28,28)
plt.figure(figsize=(7,7))
plt.imshow(img, cmap='viridis')
plt.title(f"El número escrito es: {y[i]}")
for r in range(28):
    for c in range(28):
        plt.text(c, r, str(int(img[r,c]*255)), ha='center', va='center', color='white', fontsize=5)
plt.axis('off'); plt.show()

fig, axes = plt.subplots(2,5, figsize=(10,5))
for j, ax in enumerate(axes.flat):
    ax.imshow(X[j].reshape(28,28), cmap='viridis')
    ax.set_title(f"{y[j]}", fontsize=12)
    ax.axis('off')
plt.suptitle("Primeros 10 dígitos del dataset", fontsize=14)
plt.show()

# -----------------------------
# 10) ERRORES EN CUADRÍCULA (educativo)
# -----------------------------
P_all, _ = forward(X)
y_hat = P_all.argmax(axis=1)
err_idx = np.where(y != y_hat)[0]
print("Errores totales en el dataset:", len(err_idx))

if len(err_idx) > 0:
    k = min(12, len(err_idx))
    sel = err_idx[:k]
    fig, axes = plt.subplots(3,4, figsize=(12,9))
    axes = axes.flatten()
    for ax, idx in zip(axes, sel):
        ax.imshow(x.iloc[idx].values.reshape(28,28), cmap='viridis')
        ax.set_title(f"Real={y[idx]}, Pred={y_hat[idx]}")
        ax.axis('off')
    plt.suptitle("Ejemplos de dígitos mal clasificados", fontsize=16)
    plt.tight_layout()
    plt.show()

# (Opcional) curva de pérdidas
plt.figure()
plt.plot(hist_train, label='train')
plt.plot(hist_val, label='val')
plt.xlabel('Época'); plt.ylabel('Cross-Entropy'); plt.legend(); plt.title('Evolución de la pérdida')
plt.show()
