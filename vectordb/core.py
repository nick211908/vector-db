"""
A minimal vector database built from scratch, using only numpy.
 
Core ideas this implements (the same ones real vector DBs like FAISS,
Pinecone, Weaviate, Qdrant build on top of):
 
1. Storage: vectors live in one contiguous numpy array (fast, cache-friendly).
   id -> row index is kept in a dict for O(1) lookup.
2. Similarity search: brute force = compare query against every row at once
   using matrix multiplication instead of a python loop (this is what makes
   numpy fast -- it drops into vectorized C/BLAS code).
3. Distance metrics: cosine similarity and euclidean distance.
4. Persistence: vectors saved as .npy, metadata as .json.
5. Approximate Nearest Neighbor (ANN) via Locality-Sensitive Hashing (LSH):
   once you have millions of vectors, brute force (O(n) per query) gets slow.
   LSH buckets similar vectors together using random hyperplanes, so a query
   only has to compare against its own bucket instead of everything.
   This is a toy version of what HNSW/IVF indexes do in production DBs.
"""
 
from __future__ import annotations
import numpy as np
from typing import Any, Optional
from enum import Enum

from . import persistence

class Metric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"


    
class VectorDB:
    def __init__(self, dim: int, metric: Metric=Metric.COSINE):
        """
        dim: dimensionality of vectors you'll store (must be fixed and known upfront)
        metric: "cosine" or "euclidean"
        """
        if metric not in (Metric.COSINE, Metric.EUCLIDEAN):
            raise ValueError("metric must be 'cosine' or 'euclidean'")
 
        self.dim = dim
        self.metric = Metric(metric)
 
        # Vectors stored as rows of a single numpy array. We over-allocate
        # and grow geometrically (like a python list / C++ vector) instead
        # of reallocating on every insert.
        self._capacity = 16
        self._size = 0
        self._vectors = np.zeros((self._capacity, dim), dtype=np.float32)
 
        self._ids: list[str] = []          # row index -> id
        self._id_to_row: dict[str, int] = {}  # id -> row index
        self._metadata: dict[str, Any] = {}   # id -> metadata dict
 
        # Precomputed norms for cosine similarity (avoids recomputing on
        # every search -- classic space/time tradeoff).
        self._norms = np.zeros(self._capacity, dtype=np.float32)
 
    # ---------- internal helpers ----------
 
    def _grow(self):
        new_capacity = self._capacity * 2
        new_vectors = np.zeros((new_capacity, self.dim), dtype=np.float32)
        new_vectors[: self._capacity] = self._vectors
        self._vectors = new_vectors
 
        new_norms = np.zeros(new_capacity, dtype=np.float32)
        new_norms[: self._capacity] = self._norms
        self._norms = new_norms
 
        self._capacity = new_capacity
 
    # ---------- public API ----------
 
    def add(self, id: str, vector: np.ndarray, metadata: Optional[dict] = None):
        if id in self._id_to_row:
            raise ValueError(f"id '{id}' already exists, use update() or delete() it first")
 
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
 
        if self._size >= self._capacity:
            self._grow()
 
        row = self._size
        self._vectors[row] = vector
        self._norms[row] = np.linalg.norm(vector) or 1e-10  # avoid div by zero
        self._ids.append(id)
        self._id_to_row[id] = row
        self._metadata[id] = metadata or {}
        self._size += 1
 
    def add_batch(self, ids: list[str], vectors: np.ndarray, metadatas: Optional[list[dict]] = None):
        metadatas = metadatas or [None] * len(ids)
        for id, vec, meta in zip(ids, vectors, metadatas):
            self.add(id, vec, meta)
 
    def update(self, id: str, vector: Optional[np.ndarray] = None, metadata: Optional[dict] = None):
        """
        Updates an existing id in place (row index and insertion order stay
        the same). Pass only `vector`, only `metadata`, or both.
        """
        if id not in self._id_to_row:
            raise KeyError(f"id '{id}' not found")

        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32)
            if vector.shape != (self.dim,):
                raise ValueError(f"expected vector of shape ({self.dim},), got {vector.shape}")
            row = self._id_to_row[id]
            self._vectors[row] = vector
            self._norms[row] = np.linalg.norm(vector) or 1e-10

        if metadata is not None:
            self._metadata[id] = metadata

    def delete(self, id: str):
        """
        Deletes by swapping the last row into the deleted row's place
        (O(1) instead of shifting the whole array -- same trick used in
        many array-backed structures).
        """
        if id not in self._id_to_row:
            raise KeyError(f"id '{id}' not found")
 
        row = self._id_to_row[id]
        last_row = self._size - 1
        last_id = self._ids[last_row]
 
        self._vectors[row] = self._vectors[last_row]
        self._norms[row] = self._norms[last_row]
        self._ids[row] = last_id
        self._id_to_row[last_id] = row
 
        self._ids.pop()
        del self._id_to_row[id]
        del self._metadata[id]
        self._size -= 1
 
    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float, dict]]:
        """
        Brute-force search: compare query against every stored vector at once.
        Returns [(id, score, metadata), ...] sorted best-first.
        For cosine: score = similarity (higher is better, range -1 to 1).
        For euclidean: score = distance (lower is better).
        """
        if self._size == 0:
            return []
 
        query = np.asarray(query, dtype=np.float32)
        active = self._vectors[: self._size]  # only the filled rows
 
        if self.metric == "cosine":
            query_norm = np.linalg.norm(query) or 1e-10
            # dot product of query with every row = matrix-vector multiply
            dots = active @ query
            scores = dots / (self._norms[: self._size] * query_norm)
            order = np.argsort(-scores)[:k]  # descending
        else:  # euclidean
            diffs = active - query
            dists = np.linalg.norm(diffs, axis=1)
            scores = dists
            order = np.argsort(scores)[:k]  # ascending
 
        results = []
        for row in order:
            id = self._ids[row]
            results.append((id, float(scores[row]), self._metadata[id]))
        return results
 
    def get(self, id: str) -> tuple[np.ndarray, dict]:
        row = self._id_to_row[id]
        return self._vectors[row].copy(), self._metadata[id]
 
    def __len__(self):
        return self._size
 
    # ---------- persistence ----------
 
    def save(self, path: str):
        """Saves to `<path>.npy` (vectors) and `<path>.json` (ids + metadata)."""
        persistence.save_to_disk(
            path,
            self._vectors[: self._size],
            self.dim,
            self.metric.value,
            self._ids,
            self._metadata,
        )

    @classmethod
    def load(cls, path: str) -> "VectorDB":
        vectors, dim, metric, ids, metadata = persistence.load_from_disk(path)
        db = cls(dim=dim, metric=metric)
        db.add_batch(ids, vectors, [metadata[id] for id in ids])
        return db
