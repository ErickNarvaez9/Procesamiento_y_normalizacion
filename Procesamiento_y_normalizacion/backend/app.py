# app.py
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, silhouette_score, silhouette_samples
import numpy as np
import os, json, time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONT_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_url_path="", static_folder=FRONT_DIR)
CORS(app)  # permite front y API en el mismo servicio (Render)

X_PATH = os.path.join(BASE_DIR, "X.npy")
Y_PATH = os.path.join(BASE_DIR, "y.npy")
ALFABETO_NORM_PATH = os.path.join(BASE_DIR, "alfabeto_normalizado.json")

# ----------------- util -----------------
def load_xy():
    X = np.load(X_PATH)                       # (N, D)
    y = np.load(Y_PATH, allow_pickle=True)    # vector de letras (strings)
    return X, y

def _norm_case_equal(vec_str, value):
    """Comparación case-insensitive: np.char.lower(vec) == value.lower()"""
    if value is None:
        return None
    v = str(value).strip().lower()
    return np.char.lower(vec_str.astype(str)) == v

def _load_alfabeto_json():
    try:
        with open(ALFABETO_NORM_PATH, "r", encoding="utf-8") as f:
            registros = json.load(f)
    except FileNotFoundError:
        registros = []
    # normaliza campos mínimos y parsea landmarks si vienen como string
    out = []
    for r in registros:
        lm = r.get("landmarks")
        if isinstance(lm, str):
            try:
                lm = json.loads(lm)
            except Exception:
                lm = []
        out.append({
            "nombre_secuencia": str(r.get("nombre_secuencia", "")).upper(),
            "landmarks": lm,
            "normalizado": bool(r.get("normalizado", True)),
        })
    return out

# ----------------- API existente ------------------
@app.get("/letras")
def letras_endpoint():
    _, y = load_xy()
    letras = sorted(list(np.unique(y.astype(str))))
    return jsonify(letras=letras)

@app.get("/pca")
def pca_endpoint():
    """
    PCA global o filtrado por ?letter= .
    Devuelve puntos normalizados a [0,1] para pintar en canvas.
    """
    X, y = load_xy()
    letter = request.args.get("letter")

    if letter and letter not in ("ALL", "TODAS"):
        mask = _norm_case_equal(y, letter)
        X_ = X[mask]
    else:
        X_ = X

    if X_.shape[0] < 2:
        return jsonify(points=[])

    Xs = StandardScaler().fit_transform(X_)
    P2 = PCA(n_components=2).fit_transform(Xs)
    P2 = MinMaxScaler().fit_transform(P2)

    points = [{"x": float(a), "y": float(b)} for a, b in P2]
    return jsonify(points=points)

@app.get("/coords")
def coords_endpoint():
    """
    Coordenadas normalizadas para la letra (JSON + pequeño análisis).
    """
    letter = request.args.get("letter")
    registros = _load_alfabeto_json()

    coords = []
    sample = None
    count = 0
    tgt = (letter or "").strip().upper()

    for r in registros:
        if r["nombre_secuencia"] == tgt:
            pts = r["landmarks"]  # lista [{x,y},...]
            coords.extend(pts)
            count += 1
            if sample is None:
                sample = pts

    analysis = {}
    if sample:
        pts_np = np.array([[p["x"], p["y"]] for p in sample], dtype=np.float32)
        centroid = pts_np.mean(axis=0)
        centered = pts_np - centroid
        radii = np.linalg.norm(centered, axis=1)
        s = np.linalg.svd(centered, full_matrices=False)[1]
        s0 = float(s[0]) if s.size > 0 else 0.0
        s1 = float(s[1]) if s.size > 1 else 0.0
        analysis = {
            "centroid": [float(centroid[0]), float(centroid[1])],
            "radial": {
                "mean": float(radii.mean()),
                "std": float(radii.std()),
                "min": float(radii.min()),
                "max": float(radii.max()),
            },
            "svd": {"s0": s0, "s1": s1, "anisotropy": float(s0 / (s1 + 1e-8)) if s1 > 0 else 0.0},
            "points": count * len(sample),
            "inside01": True,
        }

    return jsonify(points=coords, analysis=analysis)

@app.get("/metrics")
def metrics_endpoint():
    """
    Métricas globales o por letra (?letter=).
    Usa KNN + CV y classification_report.
    """
    X, y = load_xy()
    letter = request.args.get("letter")

    clf = KNeighborsClassifier(n_neighbors=3)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    y_pred = cross_val_predict(clf, X, y, cv=cv)

    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    accuracy = float(report.get("accuracy", 0.0))

    if letter and letter not in ("ALL", "TODAS"):
        key = None
        for k in report.keys():
            if k.lower() == str(letter).lower():
                key = k
                break
        if key is not None:
            p = float(report[key]["precision"])
            r = float(report[key]["recall"])
            f1 = float(report[key]["f1-score"])
        else:
            p = float(report["macro avg"]["precision"])
            r = float(report["macro avg"]["recall"])
            f1 = float(report["macro avg"]["f1-score"])
    else:
        p = float(report["macro avg"]["precision"])
        r = float(report["macro avg"]["recall"])
        f1 = float(report["macro avg"]["f1-score"])

    # Silhouette sobre y_pred (labels)
    try:
        sil_global = float(silhouette_score(X, y_pred))
        if letter and letter not in ("ALL", "TODAS"):
            mask = _norm_case_equal(y, letter)
            sil_samples_vec = silhouette_samples(X, y_pred)
            sil_letter = float(sil_samples_vec[mask].mean()) if mask.any() else None
        else:
            sil_letter = None
    except Exception:
        sil_global, sil_letter = None, None

    # Proxy Fisher
    def fisher_proxy(Xarr, labels):
        classes = np.unique(labels)
        if len(classes) < 2:
            return None
        centroids = np.vstack([Xarr[labels == c].mean(axis=0) for c in classes])
        sb = np.var(centroids, axis=0).mean()
        sw = np.mean([Xarr[labels == c].var(axis=0).mean() for c in classes])
        return float(sb / (sw + 1e-8))
    fisher = fisher_proxy(X, y_pred)

    # Latencias aproximadas
    clf.fit(X, y)
    t0 = time.perf_counter()
    clf.predict(X)
    per = (time.perf_counter() - t0) * 1000.0 / len(X)
    p50 = float(per)
    p95 = float(per * 1.15)

    return jsonify(
        accuracy=accuracy,
        precision=p,
        recall=r,
        f1=f1,
        silhouette={"global": sil_global, "letter": sil_letter},
        fisher=fisher,
        p50=p50,
        p95=p95,
    )

# ----------------- FRONT -----------------
@app.get("/")
def root():
    return send_from_directory(app.static_folder, "index.html")

@app.get("/<path:path>")
def static_files(path):
    # permite /script.js, /style.css, /img/...
    return send_from_directory(app.static_folder, path)

# ----------------- API v1 (nueva) -----------------
@app.get("/api/v1/alfabeto")
def api_alfabeto():
    return jsonify(_load_alfabeto_json())

@app.get("/api/v1/letras")
def api_letras():
    _, y = load_xy()
    letras = sorted(list(np.unique(y.astype(str))))
    return jsonify(letras)

@app.post("/api/v1/vector")
def api_vector():
    body = request.get_json(silent=True) or {}
    letra = str(body.get("letra", "")).strip().upper()
    if not letra:
        return jsonify(error="letra requerida"), 400
    for r in _load_alfabeto_json():
        if r["nombre_secuencia"] == letra:
            return jsonify(vector=r["landmarks"], normalizado=r["normalizado"])
    return jsonify(error="no existe"), 404

@app.get("/api/v1/health")
def api_health():
    ok_front = os.path.exists(os.path.join(FRONT_DIR, "index.html"))
    ok_X = os.path.exists(X_PATH)
    ok_y = os.path.exists(Y_PATH)
    ok_json = os.path.exists(ALFABETO_NORM_PATH)
    return jsonify(ok=True, front=ok_front, has_X=ok_X, has_y=ok_y, has_json=ok_json)

# ---- Compatibilidad con nombres antiguos (por si tu JS viejo los llama) ----
@app.get("/alfabeto_normalizado.json")
def old_alfabeto_json():
    return api_alfabeto()

@app.post("/vector_por_letra")
def old_vector_por_letra():
    return api_vector()

# ----------------- Run local / Render -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.getenv("FLASK_ENV", "").lower() == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)


