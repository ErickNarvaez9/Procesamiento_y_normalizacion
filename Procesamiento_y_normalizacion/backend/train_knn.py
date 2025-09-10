import json, time
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, silhouette_score
)

ROOT = Path(__file__).resolve().parent
X = np.load(ROOT / "X.npy")
y = np.load(ROOT / "y.npy")

# ---- resumen de clases ----
cnt = Counter(y.tolist())
classes = np.array(list(cnt.keys()))
counts = np.array(list(cnt.values()))

classes_ok = classes[counts >= 2]         # clases con >=2 muestras (válidas para split estratificado)
classes_1  = classes[counts == 1]         # clases con 1 muestra
idx_ok     = np.isin(y, classes_ok)       # índices de muestras válidas
idx_single = np.isin(y, classes_1)        # índices de las de una sola muestra (no entran al test)

X_ok, y_ok = X[idx_ok], y[idx_ok]

warn = {
    "total_samples": int(len(y)),
    "total_classes": int(len(classes)),
    "classes_ok_count": int(len(classes_ok)),
    "classes_single_count": int(len(classes_1)),
    "classes_single": [str(c) for c in classes_1]
}

# --- split seguro ---
if len(classes_ok) >= 2 and len(X_ok) >= 4:
    # estratificado solo con clases válidas
    Xtr, Xte, ytr, yte = train_test_split(
        X_ok, y_ok, test_size=0.2, stratify=y_ok, random_state=42
    )
    # opcional: también puedes añadir al train las singletons para no perderlas
    if idx_single.any():
        Xtr = np.vstack([Xtr, X[idx_single]])
        ytr = np.concatenate([ytr, y[idx_single]])
else:
    # Si no hay suficientes clases válidas, haz split simple (no estratificado)
    # y deja todo lo demás en train.
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    warn["fallback_no_stratify"] = True

# --- KNN ---
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(Xtr, ytr)
yp = knn.predict(Xte)

acc = accuracy_score(yte, yp)
prec = precision_score(yte, yp, average="macro", zero_division=0)
rec = recall_score(yte, yp, average="macro", zero_division=0)
f1 = f1_score(yte, yp, average="macro", zero_division=0)

# --- Silhouette (si hay >=2 clases en todo el dataset) ---
if len(np.unique(y)) > 1:
    sil = float(silhouette_score(X, y, metric='euclidean'))
else:
    sil = None

# --- Fisher global ---
classes_all = np.unique(y)
mu = X.mean(axis=0)
varw = 0.0; varb = 0.0
for c in classes_all:
    Xi = X[y==c]; mu_i = Xi.mean(axis=0)
    varw += Xi.var(axis=0, ddof=1).sum()
    varb += ((mu_i - mu)**2).sum() * len(Xi)
fisher = float(varb / (varw + 1e-8))

# --- Latencias RAM (top-k) ---
def topk_ram(q, k=5):
    d = np.linalg.norm(X - q, axis=1)
    idx = np.argpartition(d, k)[:k]
    idx = idx[np.argsort(d[idx])]
    return idx, d[idx]

dur = []
rng = np.random.default_rng(0)
iters = min(1000, X.shape[0])
for _ in range(iters):
    i = rng.integers(0, X.shape[0])
    q = X[i]
    t0 = time.perf_counter()
    topk_ram(q, k=5)
    t1 = time.perf_counter()
    dur.append((t1 - t0) * 1000.0)

p50 = float(np.percentile(dur, 50))
p95 = float(np.percentile(dur, 95))

metrics = {
    "accuracy": float(acc),
    "precision_macro": float(prec),
    "recall_macro": float(rec),
    "f1_macro": float(f1),
    "silhouette": sil,
    "fisher_global": fisher,
    "latency_ms_p50": p50,
    "latency_ms_p95": p95,
    "samples": int(X.shape[0]),
    "dim": int(X.shape[1]),
    "split_info": warn
}

with open(ROOT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)

print("[OK] metrics.json")
for k, v in metrics.items():
    print(f" - {k}: {v}")
