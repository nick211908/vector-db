# Build-Your-Own Vector Database — Implementation Plan

This document is a step-by-step plan for building a from-scratch, pure-Python/NumPy vector database, structured as a proper project you can grow over time. It uses the working reference implementation (`vector_db.py` / `demo.py`) already built as the starting point, and lays out how to organize it, test it, and extend it.

## 1. Goal and scope

Build a small, dependency-light vector store that supports:

- adding vectors with metadata
- upserting and deleting by id
- exact nearest-neighbor search (cosine similarity or L2 distance)
- metadata filtering during search
- persistence to disk (save/load)

This is a **flat index** (brute-force O(n) search per query). That's the right choice for datasets up to roughly 100k–500k vectors on a single machine, and it always returns exact — not approximate — results. The plan below is written so the internals can later be swapped for an approximate nearest-neighbor (ANN) index without changing the public API.

## 2. Folder structure

Organize the project like this:

```
vector-db/
├── README.md
├── requirements.txt
├── pyproject.toml               # optional, if you want it pip-installable
├── vectordb/
│   ├── __init__.py              # exposes VectorDB, SearchResult
│   ├── core.py                  # the VectorDB class (from vector_db.py)
│   ├── result.py                # SearchResult dataclass
│   └── persistence.py           # save/load helpers (npz + json)
├── examples/
│   └── demo.py                  # the walkthrough script (already built)
├── tests/
│   ├── test_add_delete.py
│   ├── test_search.py
│   ├── test_filter.py
│   └── test_persistence.py
└── data/
    └── .gitkeep                 # saved .npz/.json databases live here (gitignored)
```

Notes on this layout:

- `vectordb/` is the actual package — this is what you'd `pip install -e .` or import from other projects.
- `core.py` holds the `VectorDB` class itself; `result.py` holds the small `SearchResult` dataclass; `persistence.py` isolates the save/load file-format logic so it can change (e.g. swap JSON for SQLite metadata) without touching search logic.
- `examples/` is for runnable scripts a human reads top-to-bottom to understand usage — keep these separate from `tests/`, which are for automated verification.
- `tests/` mirrors the feature areas: creation/mutation, search correctness, filtering, and round-trip persistence.
- `data/` is a working directory for saved databases; add it to `.gitignore` so large `.npz` files don't get committed.

If you'd rather keep things minimal (a single script, no package), you can skip the `vectordb/` split and just keep `vector_db.py` + `demo.py` at the root — the step-by-step guide below covers both options.

## 3. Step-by-step implementation guide

### Step 1 — Set up the project

```bash
mkdir vector-db && cd vector-db
python3 -m venv .venv
source .venv/bin/activate       # or .venv\Scripts\activate on Windows
pip install numpy
```

Create `requirements.txt`:

```
numpy>=1.24
```

Create `.gitignore`:

```
.venv/
__pycache__/
*.npz
data/*.json
!data/.gitkeep
```

### Step 2 — Define the data model (`vectordb/result.py`)

Start with the smallest building block: what a single search hit looks like. This is a dataclass with `id`, `score`, `vector`, and `metadata`. Keeping it in its own file means both `core.py` and any future re-ranking or export code can import it without circular dependencies.

### Step 3 — Build core storage (`vectordb/core.py`, part 1)

Implement the `VectorDB.__init__` and internal storage:

- a single 2D NumPy array (`_vectors`) holding all vectors, so search can be vectorized
- a parallel list of ids (`_ids`) and metadata dicts (`_metadata`)
- a dict `_id_to_row` mapping id → row index for O(1) lookup

Decide the constructor's contract up front: `dim` can be passed explicitly, or inferred from the first vector added. `metric` is `"cosine"` or `"l2"`, chosen once at construction time.

### Step 4 — Implement mutation (`add`, `add_many`, `delete`, `get`)

Work through these in order, since each builds on the last:

1. `add(id, vector, metadata)` — validate the vector's dimensionality, append to `_vectors` (or replace the row if the id already exists — this makes `add` double as an upsert), append id/metadata, update `_id_to_row`.
2. `add_many(items)` — a thin loop over `add`, accepting `(vector,)`, `(vector, metadata)`, or `(id, vector, metadata)` tuples for convenience during bulk loading.
3. `delete(id)` — remove the row from `_vectors` with `np.delete`, drop the id/metadata, and **re-index every row after the deleted one** in `_id_to_row` (this is the easiest bug to introduce — write the test for it immediately, see Step 8).
4. `get(id)` — simple lookup by id, useful for debugging and for confirming an upsert worked.

### Step 5 — Implement search (`vectordb/core.py`, part 2)

This is the core value of the whole project:

1. Convert the query to a row vector, validate its dimension matches `self.dim`.
2. If a `filter` was passed, narrow the candidate rows first (dict → equality match on metadata; callable → arbitrary predicate).
3. Compute similarity in one vectorized operation over the candidate matrix:
   - cosine: `(matrix @ query) / (||matrix rows|| * ||query||)`, with a small epsilon added to norms to avoid divide-by-zero
   - L2: `-||matrix rows - query||`, negated so "higher score = better match" stays consistent across both metrics
4. Use `np.argpartition` to get the top-k indices in O(n) rather than fully sorting all candidates, then sort just those k for the final order.
5. Wrap each result as a `SearchResult`.

Write this function assuming zero results is a valid, expected case (empty DB, or a filter that matches nothing) — return `[]` rather than raising.

### Step 6 — Implement persistence (`vectordb/persistence.py`)

Split the file format from the class logic:

- `save(path)`: write vectors to `{path}.npz` via `np.savez_compressed` (compression matters once you have more than a few thousand vectors), and write `{dim, metric, ids, metadata}` to `{path}.json`.
- `load(path)`: read both files back, reconstruct `_id_to_row` from the loaded `ids` list, and return a fully-populated `VectorDB` instance.

Keep the two files paired by convention (same base path, `.npz` + `.json` suffixes) so a "database" is really just two files sitting next to each other — easy to copy, move, or check into a data bucket.

### Step 7 — Wire up the package (`vectordb/__init__.py`)

```python
from .core import VectorDB
from .result import SearchResult

__all__ = ["VectorDB", "SearchResult"]
```

This lets callers do `from vectordb import VectorDB` instead of reaching into `vectordb.core`.

### Step 8 — Write tests

Cover these cases — each maps to a real bug class this kind of data structure invites:

- **`test_add_delete.py`**: add several vectors, delete one from the middle, confirm the remaining ids still map to their correct vectors (this is the re-indexing bug from Step 4).
- **`test_search.py`**: known vectors with an obvious "nearest" answer (e.g. orthogonal unit vectors), assert the top result is correct and scores are ordered descending.
- **`test_filter.py`**: add vectors across two metadata categories, confirm a filtered search only returns the matching category, and confirm an empty-matching filter returns `[]` rather than erroring.
- **`test_persistence.py`**: build a DB, save it, load it into a new instance, and assert search results are identical before and after the round trip.

Run with:

```bash
pip install pytest
pytest tests/ -v
```

### Step 9 — Keep the demo as living documentation

Move `demo.py` into `examples/` and keep it in sync with the API — it's the fastest way for you (or anyone else) to remember how the pieces fit together six months from now. Run it as a smoke test whenever you change the core API:

```bash
python examples/demo.py
```

### Step 10 — (Optional) Package it properly

If you want to `pip install` this into other projects, add a minimal `pyproject.toml`:

```toml
[project]
name = "vectordb"
version = "0.1.0"
dependencies = ["numpy>=1.24"]

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

Then `pip install -e .` from the project root makes `import vectordb` work from anywhere in your environment.

## 4. Suggested build order (checklist)

- [ ] Scaffold folders and virtual environment (Step 1)
- [ ] `SearchResult` dataclass (Step 2)
- [ ] `VectorDB.__init__` + internal storage (Step 3)
- [ ] `add` / `add_many` / `delete` / `get` (Step 4)
- [ ] `search` with cosine + L2 + filtering (Step 5)
- [ ] `save` / `load` (Step 6)
- [ ] Package `__init__.py` (Step 7)
- [ ] Tests for all four areas (Step 8)
- [ ] Move/verify demo script (Step 9)
- [ ] Optional: packaging (Step 10)

## 5. Where to go next

Once this is working end to end, natural extensions — each a self-contained follow-up project rather than something to bolt on right away — include:

- **Approximate search at scale**: swap the brute-force loop for an ANN index (e.g. FAISS `IndexHNSWFlat` or `IndexIVFFlat`) once you're past a few hundred thousand vectors and exact search gets too slow. Keep the same `VectorDB` public API so calling code doesn't need to change.
- **Metadata-rich filtering**: replace the simple equality/predicate filter with a small query language (range filters, `$in`, `$and`/`$or`) if metadata filtering needs grow.
- **Text embeddings built in**: add an optional embedding step (e.g. `sentence-transformers`) so the DB can accept raw text and embed it internally, rather than requiring pre-computed vectors.
- **Concurrent access**: if multiple processes need to read/write the same database, move from flat-file persistence to something like SQLite (metadata) + a memory-mapped vector file, or a lightweight server process in front of the `VectorDB` instance.
- **Batch queries**: extend `search` to accept a matrix of queries at once and return results per row, which is more efficient than looping one query at a time.