"""
File-format logic for saving/loading a VectorDB to disk, kept separate from
core.py so the on-disk format can change (e.g. swap JSON metadata for
SQLite) without touching storage or search logic.

A saved database is two files sharing a base path:
  <path>.npy   -- the vectors, written with np.save
  <path>.json  -- dim, metric, ids, and metadata
"""

from __future__ import annotations
import json
import os
import numpy as np
from typing import Any


def save_to_disk(
    path: str,
    vectors: np.ndarray,
    dim: int,
    metric: str,
    ids: list[str],
    metadata: dict[str, Any],
) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path + ".npy", vectors)
    with open(path + ".json", "w") as f:
        json.dump(
            {
                "dim": dim,
                "metric": metric,
                "ids": ids,
                "metadata": metadata,
            },
            f,
        )


def load_from_disk(path: str) -> tuple[np.ndarray, int, str, list[str], dict[str, Any]]:
    with open(path + ".json") as f:
        data = json.load(f)
    vectors = np.load(path + ".npy")
    return vectors, data["dim"], data["metric"], data["ids"], data["metadata"]
