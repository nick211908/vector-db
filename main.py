import time

import numpy as np

from vectordb.core import VectorDB, Metric


def basic_demo():
    """The original small, readable API walkthrough."""
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
    print("basic search:", results)

    # filter before ranking: dict (equality on all keys) or a predicate callable
    print(db.search(np.array([1, 0, 0]), k=2, where={"label": "x-axis"}))
    print(db.search(np.array([1, 0, 0]), k=2, where=lambda m: m["label"].startswith("x")))

    db.update("a", metadata={"label": "x-axis (renamed)"})
    db.delete("b")

    db.save("data/my_db")  # writes data/my_db.npy + data/my_db.json
    db2 = VectorDB.load("data/my_db")
    print("reloaded:", db2.get("a"))


def _recall(approx, exact):
    """Fraction of the exact top-k ids that the approximate search also found."""
    exact_ids = {id for id, _, _ in exact}
    if not exact_ids:
        return 1.0
    hits = sum(1 for id, _, _ in approx if id in exact_ids)
    return hits / len(exact_ids)


def _make_clustered(rng, n, dim, n_clusters, spread=0.15):
    """
    Synthetic vectors with real cluster structure: pick `n_clusters` random
    centers, then scatter points tightly around them.

    This matters for the benchmark. IVF relies on the data actually forming
    clusters for k-means to carve up -- that's true of real embeddings (text,
    images), which live on a low-dimensional manifold. *Uniform* random vectors
    in high dim have no such structure (every point is ~equidistant from every
    other -- the curse of dimensionality), so k-means finds nothing to grab and
    IVF recall looks artificially terrible. Clustered data is the honest test.
    """
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    which = rng.integers(0, n_clusters, size=n)
    data = centers[which] + spread * rng.standard_normal((n, dim)).astype(np.float32)
    return data.astype(np.float32)


def ivf_benchmark(n=20_000, dim=128, queries=200, k=10, seed=0):
    """
    Build the same vectors into a flat (brute-force) DB and an IVF index, then
    compare them on the two things that matter: recall (does IVF find the same
    neighbors?) and speed (is it actually faster?).

    IVF trades exactness for speed -- it only scans the `nprobe` nearest cells
    instead of all N vectors. So we expect recall < 1.0 that climbs toward 1.0
    as nprobe grows, and query time well below brute force at low nprobe.
    """
    rng = np.random.default_rng(seed)
    # clustered data, not uniform -- see _make_clustered for why it matters
    data = _make_clustered(rng, n, dim, n_clusters=100)
    q_idx = rng.choice(n, size=queries, replace=False)
    # queries = existing points nudged slightly, so each has real near-neighbors
    q = data[q_idx] + 0.05 * rng.standard_normal((queries, dim)).astype(np.float32)
    q = q.astype(np.float32)

    db = VectorDB(dim=dim, metric=Metric.COSINE)
    db.add_batch([str(i) for i in range(n)], data)
    print(f"\nIVF benchmark: {n} vectors, dim={dim}, k={k}, {queries} queries")

    # exact brute-force baseline
    t0 = time.perf_counter()
    exact = [db.search(query, k=k) for query in q]
    exact_ms = (time.perf_counter() - t0) / queries * 1000
    print(f"  brute force : {exact_ms:6.2f} ms/query   recall 1.000 (baseline)")

    # nlist ~ sqrt(n) is the usual rule of thumb; sweep nprobe to show the knob
    nlist = int(np.sqrt(n))
    for nprobe in (1, 4, 8, 16, 32):
        if nprobe > nlist:
            break
        db.build_index(nlist=nlist, nprobe=nprobe, seed=seed)
        t0 = time.perf_counter()
        approx = [db.search(query, k=k, use_index=True) for query in q]
        ivf_ms = (time.perf_counter() - t0) / queries * 1000
        recall = np.mean([_recall(a, e) for a, e in zip(approx, exact)])
        speedup = exact_ms / ivf_ms if ivf_ms else float("inf")
        print(
            f"  IVF nprobe={nprobe:<2d}: {ivf_ms:6.2f} ms/query   "
            f"recall {recall:5.3f}   {speedup:4.1f}x faster"
        )


if __name__ == "__main__":
    basic_demo()
    ivf_benchmark()
