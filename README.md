# vectordb

A minimal vector database built from scratch with only NumPy — no FAISS, no
external ANN library. It's a **flat index**: brute-force O(n) search per
query, exact (not approximate) results. That's the right tradeoff up to
roughly 100k–500k vectors on a single machine.

## Why

Real vector DBs (FAISS, Pinecone, Weaviate, Qdrant) are built on the same
handful of ideas. This project implements them directly so the mechanics are
visible instead of hidden behind a library:

1. **Storage** — vectors live in one contiguous NumPy array (`_vectors`),
   which keeps search cache-friendly and vectorizable. `id -> row index` is
   a dict for O(1) lookup.
2. **Search** — brute force means comparing the query against every row at
   once via matrix multiplication, instead of a Python loop over rows. This
   is what makes NumPy fast: it drops into vectorized C/BLAS code.
3. **Metrics** — cosine similarity and Euclidean (L2) distance.
4. **Mutation** — deletes swap the last row into the deleted slot (O(1))
   instead of shifting the whole array.
5. **Persistence** — vectors saved as `.npy`, metadata/ids/config as `.json`.

## Install

Managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Requires Python >=3.13 and numpy>=2.5.0 (see `pyproject.toml`).

## Usage

```python
from vectordb.core import VectorDB, Metric
import numpy as np

db = VectorDB(dim=3, metric=Metric.COSINE)  # or metric="cosine"

db.add("a", np.array([1, 0, 0]), metadata={"label": "x-axis"})
db.add("b", np.array([0, 1, 0]), metadata={"label": "y-axis"})
db.add_batch(
    ids=["c", "d"],
    vectors=np.array([[0, 0, 1], [1, 1, 0]]),
    metadatas=[{"label": "z-axis"}, {"label": "diagonal"}],
)

results = db.search(np.array([1, 0, 0]), k=2)
# [(id, score, metadata), ...] sorted best-first
# cosine: higher score = better match (range -1 to 1)
# euclidean: lower score = better match

db.update("a", metadata={"label": "x-axis (renamed)"})
db.delete("b")

db.save("data/my_db")           # writes data/my_db.npy + data/my_db.json
db2 = VectorDB.load("data/my_db")
```

## API

`VectorDB(dim, metric=Metric.COSINE)`

| Method | Description |
|---|---|
| `add(id, vector, metadata=None)` | Insert a new vector. Raises `ValueError` if `id` already exists. |
| `add_batch(ids, vectors, metadatas=None)` | Insert many at once. |
| `update(id, vector=None, metadata=None)` | Update an existing id's vector and/or metadata in place. Raises `KeyError` if `id` doesn't exist. |
| `delete(id)` | Remove an id (O(1), swap-with-last-row). |
| `search(query, k=5)` | Return the `k` nearest `(id, score, metadata)` tuples. |
| `get(id)` | Return `(vector, metadata)` for a single id. |
| `save(path)` / `VectorDB.load(path)` | Persist to / restore from `<path>.npy` + `<path>.json`. |
| `len(db)` | Number of stored vectors. |

## Project layout

```
vector_db/
├── vectordb/
│   ├── core.py          # VectorDB class: storage, mutation, search
│   ├── persistence.py   # save/load file-format logic (kept separate from core)
│   ├── results.py       # SearchResult dataclass
│   └── __init__.py      # (not yet wired up — see Roadmap)
├── data/                 # saved .npy/.json databases live here
├── main.py
└── pyproject.toml
```

`persistence.py` is deliberately decoupled from `VectorDB` (plain function
signatures, no imports of `core.py`) so the on-disk format can change later
—e.g. swapping JSON metadata for SQLite — without touching storage or
search logic.

## Status

Working: add / add_batch / update / delete / get, cosine + euclidean
search, save/load round-tripping, a `Metric` string-enum for the metric
type.

Not yet implemented:

- Metadata filtering during `search()` (equality or predicate-based)
- `vectordb/__init__.py` exports (`from vectordb import VectorDB`)
- `tests/` (add/delete re-indexing, search correctness, filtering,
  persistence round-trip)
- `examples/` demo script
- Approximate nearest-neighbor search (LSH/HNSW/IVF) for datasets beyond
  what brute force can handle

## Not a goal

This isn't meant to compete with FAISS/Pinecone/Qdrant on performance or
scale — it's a from-scratch build to understand how they work underneath.
