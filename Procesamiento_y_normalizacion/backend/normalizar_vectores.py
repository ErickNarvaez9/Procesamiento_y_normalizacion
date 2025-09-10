# -*- coding: utf-8 -*-
"""
Normaliza alfabeto.json a [0,1] por muestra (bbox) y genera:
- alfabeto_normalizado.json  (coordenadas normalizadas por muestra)
- X.npy                      (N x D con vector plano x1,y1,...)
- y.npy                      (etiquetas, una letra por fila)
- vectores.csv               (auditoría en texto)
- embeddings.npy             (por defecto: rasgos "engineered" 9D; opcional: 42D)


"""

import os, json, csv
from typing import List, Dict, Any
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IN_JSON  = os.path.join(BASE_DIR, "alfabeto.json")
OUT_JSON = os.path.join(BASE_DIR, "alfabeto_normalizado.json")
OUT_X    = os.path.join(BASE_DIR, "X.npy")
OUT_y    = os.path.join(BASE_DIR, "y.npy")
OUT_CSV  = os.path.join(BASE_DIR, "vectores.csv")
OUT_EMB  = os.path.join(BASE_DIR, "embeddings.npy")

def parse_landmarks(field: Any) -> List[Dict[str, float]]:
    if isinstance(field, str):
        field = json.loads(field)
    if not isinstance(field, list):
        raise ValueError("El campo landmarks/coordenadas no es lista ni string JSON.")
    pts: List[Dict[str, float]] = []
    for p in field:
        if isinstance(p, dict):
            x = float(p.get("x", 0.0)); y = float(p.get("y", 0.0))
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            x = float(p[0]); y = float(p[1])
        else:
            raise ValueError(f"Formato de punto no soportado: {p}")
        if not np.isfinite(x): x = 0.0
        if not np.isfinite(y): y = 0.0
        pts.append({"x": x, "y": y})
    return pts

def to_xy_array(pts: List[Dict[str, float]]) -> np.ndarray:
    arr = np.array([[p["x"], p["y"]] for p in pts], dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    return arr

def normalize_by_bbox(xy: np.ndarray) -> np.ndarray:
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    rng = np.clip(mx - mn, 1e-8, None)
    xy_n = (xy - mn) / rng
    return np.clip(xy_n, 0.0, 1.0)

def engineered_features(xy_norm: np.ndarray) -> np.ndarray:
    centroid = xy_norm.mean(axis=0)
    centered = xy_norm - centroid
    r = np.linalg.norm(centered, axis=1)
    if centered.shape[0] >= 2:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        s0 = float(S[0]) if S.size >= 1 else 0.0
        s1 = float(S[1]) if S.size >= 2 else 0.0
    else:
        s0, s1 = 0.0, 0.0
    aniso = s0 / (s1 + 1e-8) if s1 > 0 else 0.0
    return np.array([
        float(centroid[0]), float(centroid[1]),
        float(r.mean()), float(r.std()),
        float(r.min() if r.size else 0.0),
        float(r.max() if r.size else 0.0),
        s0, s1, float(aniso)
    ], dtype=np.float32)

def main():
    if not os.path.exists(IN_JSON):
        raise FileNotFoundError(f"No existe {IN_JSON}")
    with open(IN_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)

    registros_out = []
    vectores_plano, etiquetas, emb_engineered = [], [], []
    n_saltos, D_esperada = 0, None

    for i, item in enumerate(raw):
        letra = str(item.get("nombre_secuencia") or item.get("letra") or "").strip().lower()
        if not letra:
            n_saltos += 1; continue
        lm_field = item.get("landmarks", item.get("coordenadas"))
        if lm_field is None:
            n_saltos += 1; continue

        try:
            pts = parse_landmarks(lm_field)
        except Exception as e:
            print(f"[WARN] Muestra {i}: landmarks inválidos → {e}")
            n_saltos += 1; continue

        xy = to_xy_array(pts)
        if xy.ndim != 2 or xy.shape[1] != 2:
            print(f"[WARN] Muestra {i}: forma {xy.shape} inválida, se espera (N,2)")
            n_saltos += 1; continue

        xy_norm = normalize_by_bbox(xy)

        coords = [{"x": float(x), "y": float(y)} for x, y in xy_norm]
        registros_out.append({
            "nombre_secuencia": letra,
            "coordenadas": coords,
            "landmarks": json.dumps(coords, ensure_ascii=False),
            "modalidad": "hand_xy",
            "version": 1
        })

        plano = xy_norm.flatten().tolist()
        vectores_plano.append(plano)
        etiquetas.append(letra)
        emb_engineered.append(engineered_features(xy_norm))
        if D_esperada is None:
            D_esperada = len(plano)

    if not vectores_plano:
        raise RuntimeError("No se generaron vectores. Revisa entradas.")

    X = np.asarray(vectores_plano, dtype=np.float32)
    y = np.asarray(etiquetas)
    EMB = np.vstack(emb_engineered).astype(np.float32)

    np.save(OUT_X, X)
    np.save(OUT_y, y)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(registros_out, f, ensure_ascii=False, indent=2)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["letra"] + [f"v{i+1}" for i in range(X.shape[1])]
        writer.writerow(header)
        for letra, vec in zip(etiquetas, X):
            writer.writerow([letra] + [f"{v:.6f}" for v in vec])

    # Guarda engineered por defecto; si prefieres 42D: np.save(OUT_EMB, X)
    np.save(OUT_EMB, EMB)

    print(f"[OK] Normalización y exportación completadas:")
    print(f"     - {OUT_JSON}")
    print(f"     - {OUT_X}        shape={X.shape}")
    print(f"     - {OUT_y}        shape={y.shape}")
    print(f"     - {OUT_EMB}      shape={EMB.shape}")
    print(f"     - {OUT_CSV}")
    if D_esperada is not None:
        print(f"     Dimensión plana detectada: D={D_esperada}")
    if n_saltos:
        print(f"[INFO] Muestras omitidas: {n_saltos}")

if __name__ == "__main__":
    main()
