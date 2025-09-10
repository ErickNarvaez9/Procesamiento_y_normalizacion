# -*- coding: utf-8 -*-
import os
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "lse_letras")
DISTANCE = os.getenv("QDRANT_DISTANCE", "Euclid")  # Euclid | Cosine | Dot

def get_client() -> QdrantClient:
    if not QDRANT_URL or not QDRANT_API_KEY:
        raise RuntimeError("QDRANT_URL o QDRANT_API_KEY no configurados.")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def _distance_obj() -> Distance:
    d = DISTANCE.lower()
    if d == "cosine": return Distance.COSINE
    if d == "dot":    return Distance.DOT
    return Distance.EUCLID  # por defecto

def ensure_collection(client: QdrantClient, dim: int) -> None:
    if COLLECTION not in [c.name for c in client.get_collections().collections]:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=_distance_obj())
        )

def reindex(client: QdrantClient, vectors: List[List[float]], payloads: List[Dict[str, Any]], ids: Optional[List[int]] = None) -> None:
    if ids is None:
        ids = list(range(len(vectors)))
    points = [
        PointStruct(id=int(i), vector=vectors[i], payload=payloads[i])
        for i in range(len(ids))
    ]
    client.upsert(collection_name=COLLECTION, points=points)
