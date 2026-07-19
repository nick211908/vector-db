# vectordb

A minimal vector database built from scratch with only NumPy — no FAISS, no
external ANN library. It ships two search paths so you can compare them
directly:

- a **flat index** — brute-force O(n) search per query, exact results (the
  right tradeoff up to roughly 100k–500k vectors on a single machine), and
- an **IVF (inverted-file) index** — approximate search that clusters the
  vectors with from-scratch k-means and scans only the nearest few cells per
  query, trading a little recall for a large speedup.

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
6. **Approximate search (IVF)** — an optional inverted-file index. k-means
   (written from scratch, `ivf.py`) learns `nlist` centroids; every vector is
   filed under its nearest one. A query scans only the `nprobe` nearest cells
   instead of all N — trading exactness for a large speedup. This is the same
   idea as FAISS's `IndexIVFFlat`. Exact search is untouched; IVF is additive
   so the two can be compared directly (see the benchmark in `main.py`).

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

# filter before ranking: dict (equality on all keys) or a predicate callable
db.search(np.array([1, 0, 0]), k=2, where={"label": "x-axis"})
db.search(np.array([1, 0, 0]), k=2, where=lambda m: m["label"].startswith("x"))

db.update("a", metadata={"label": "x-axis (renamed)"})
db.delete("b")

db.save("data/my_db")           # writes data/my_db.npy + data/my_db.json
db2 = VectorDB.load("data/my_db")
```

### Approximate search (IVF)

For larger datasets, train an IVF index once and search it instead of
scanning every vector:

```python
db.build_index(nlist=128, nprobe=4)   # k-means training; one-shot snapshot
db.search(query, k=10, use_index=True)  # scans only the nprobe nearest cells
```

`nprobe` is the recall/speed knob: higher scans more cells (better recall,
slower); `nprobe == nlist` degenerates to exact search. Any
`add`/`update`/`delete` marks the index stale — call `build_index()` again
before the next `use_index=True` search or it raises. Run `python main.py` to
see the recall-vs-speed tradeoff on clustered synthetic data.

## API

`VectorDB(dim, metric=Metric.COSINE)`

| Method | Description |
|---|---|
| `add(id, vector, metadata=None)` | Insert a new vector. Raises `ValueError` if `id` already exists. |
| `add_batch(ids, vectors, metadatas=None)` | Insert many at once. |
| `update(id, vector=None, metadata=None)` | Update an existing id's vector and/or metadata in place. Raises `KeyError` if `id` doesn't exist. |
| `delete(id)` | Remove an id (O(1), swap-with-last-row). |
| `search(query, k=5, where=None, use_index=False)` | Return the `k` nearest `(id, score, metadata)` tuples. `where` filters by metadata before ranking: a dict for equality matching, or a callable predicate. `use_index=True` uses the IVF index (approximate); requires a fresh `build_index()`. |
| `build_index(nlist=None, nprobe=1, iters=10, seed=0)` | Train an IVF index over the current vectors. `nlist` defaults to ~`sqrt(size)`. Enables `search(..., use_index=True)`. |
| `get(id)` | Return `(vector, metadata)` for a single id. |
| `save(path)` / `VectorDB.load(path)` | Persist to / restore from `<path>.npy` + `<path>.json`. |
| `len(db)` | Number of stored vectors. |

## Project layout

```
vector_db/
├── vectordb/
│   ├── core.py          # VectorDB class: storage, mutation, search
│   ├── ivf.py           # kmeans + IVFIndex (approximate search, from scratch)
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
search, metadata filtering (`where=`) during search, save/load
round-tripping, a `Metric` string-enum for the metric type, and an IVF
approximate index (`build_index()` + `search(..., use_index=True)`) with a
recall-vs-speed benchmark in `main.py`.

Not yet implemented:

- `vectordb/__init__.py` exports (`from vectordb import VectorDB`)
- `tests/` (add/delete re-indexing, search correctness, filtering,
  persistence round-trip, IVF recall)
- `examples/` demo script
- Persisting a trained IVF index to disk (currently rebuilt after `load()`)
- Other ANN indexes (LSH / HNSW) for comparison

## Not a goal

This isn't meant to compete with FAISS/Pinecone/Qdrant on performance or
scale — it's a from-scratch build to understand how they work underneath.
