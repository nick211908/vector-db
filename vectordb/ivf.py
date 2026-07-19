"""
IVF (Inverted File) index -- an approximate nearest-neighbor accelerator,
built from scratch on NumPy to sit alongside the exact brute-force search in
core.py so the two can be compared directly (recall vs. speed).

The idea, same one FAISS's IndexIVFFlat uses:

1. Train: run k-means on the stored vectors to learn `nlist` centroids. Each
   centroid owns a Voronoi cell -- the region of space closer to it than to
   any other centroid.
2. Assign: every vector is filed under its nearest centroid. This builds the
   "inverted lists": centroid index -> list of vector rows in that cell.
3. Search: instead of scanning all N vectors, find the `nprobe` centroids
   closest to the query and scan only the vectors in those cells. Fewer
   candidates -> faster query, at the cost of possibly missing a true
   neighbor that fell in an unprobed cell (that's the "approximate" part).

Brute force is O(N) per query. IVF scans roughly N * nprobe / nlist vectors,
so with nlist ~= sqrt(N) and a small nprobe it's a large constant-factor win.

Everything here works on plain row arrays and returns row indices; it has no
knowledge of ids/metadata. VectorDB owns that mapping and layers filtering on
top of the candidate rows this returns.
"""

from __future__ import annotations
import numpy as np


def _pairwise_sq_dists(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Squared euclidean distance from every point to every centroid.

    Returns an (n_points, n_centroids) matrix. Uses the identity
        ||p - c||^2 = ||p||^2 - 2 p.c + ||c||^2
    so the whole thing is one matrix multiply (points @ centroids.T) plus two
    norm vectors -- the same vectorization trick core.search() uses, extended
    from one query to a full matrix. Squared distance is enough for argmin
    (sqrt is monotonic), so we skip the sqrt.
    """
    p_sq = np.sum(points ** 2, axis=1, keepdims=True)        # (n, 1)
    c_sq = np.sum(centroids ** 2, axis=1)                     # (nlist,)
    cross = points @ centroids.T                             # (n, nlist)
    dists = p_sq - 2 * cross + c_sq                          # broadcast -> (n, nlist)
    return np.maximum(dists, 0)  # clamp tiny negatives from float rounding


def kmeans(
    vectors: np.ndarray,
    nlist: int,
    iters: int = 10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lloyd's algorithm: the classic k-means loop, in NumPy.

      init      -- pick `nlist` distinct rows as starting centroids
      repeat:
        assign  -- each point joins its nearest centroid's cluster
        update  -- each centroid moves to the mean of its members
                   (empty clusters are re-seeded to a random point so they
                   don't stay dead)

    Returns (centroids, assignments) where centroids is (nlist, dim) and
    assignments[i] is the centroid index that vector i belongs to.

    Note: for cosine similarity, callers should L2-normalize `vectors` first.
    On the unit sphere, euclidean k-means becomes "spherical" k-means, whose
    clusters group vectors by angle -- which is exactly what cosine ranks by.
    """
    n = vectors.shape[0]
    if nlist > n:
        raise ValueError(f"nlist ({nlist}) cannot exceed number of vectors ({n})")

    rng = np.random.default_rng(seed)
    # init: distinct random rows as centroids (a light-weight stand-in for
    # k-means++; fine for a from-scratch index and keeps the code readable).
    init_idx = rng.choice(n, size=nlist, replace=False)
    centroids = vectors[init_idx].copy()

    assignments = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        # assign step: nearest centroid for every point at once
        dists = _pairwise_sq_dists(vectors, centroids)
        new_assignments = np.argmin(dists, axis=1)

        # update step: each centroid = mean of the points assigned to it
        for c in range(nlist):
            members = vectors[new_assignments == c]
            if members.shape[0] > 0:
                centroids[c] = members.mean(axis=0)
            else:
                # empty cluster: re-seed to a random point so it can recover
                centroids[c] = vectors[rng.integers(n)]

        # converged: assignments stopped changing
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments

    return centroids, assignments


class IVFIndex:
    """
    An inverted-file index over a set of vector rows.

    Holds the learned centroids and, per centroid, the list of vector rows
    that fall in its cell. `search` narrows the query to the vectors in the
    `nprobe` nearest cells and returns those candidate rows for the caller to
    score exactly.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        nlist: int,
        nprobe: int = 1,
        iters: int = 10,
        normalize: bool = False,
        seed: int = 0,
    ):
        """
        vectors: (n, dim) the rows to index (a snapshot at build time).
        nlist: number of centroids / cells.
        nprobe: how many nearest cells to scan per query (1..nlist). Higher
                nprobe -> better recall, slower search. This is *the* IVF knob.
        normalize: if True, cluster on L2-normalized vectors (spherical
                   k-means) -- use this for cosine similarity.
        """
        if not 1 <= nprobe <= nlist:
            raise ValueError(f"nprobe must be in 1..nlist ({nlist}), got {nprobe}")

        self.nlist = nlist
        self.nprobe = nprobe
        self.normalize = normalize

        train_vectors = self._maybe_normalize(vectors)
        self.centroids, assignments = kmeans(train_vectors, nlist, iters, seed)

        # inverted lists: centroid index -> array of vector rows in that cell
        self.lists: list[np.ndarray] = [
            np.where(assignments == c)[0] for c in range(nlist)
        ]

    def _maybe_normalize(self, vectors: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return vectors / norms

    def candidates(self, query: np.ndarray) -> np.ndarray:
        """
        Return the candidate vector rows for `query`: the union of the
        inverted lists belonging to the `nprobe` centroids nearest the query.

        This is the whole speedup -- the caller scores only these rows exactly
        instead of all N.
        """
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if self.normalize:
            query = self._maybe_normalize(query)

        # distance from the query to every centroid, pick the nprobe closest
        dists = _pairwise_sq_dists(query, self.centroids)[0]  # (nlist,)
        probe = np.argsort(dists)[: self.nprobe]

        parts = [self.lists[c] for c in probe if self.lists[c].size > 0]
        if not parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(parts)
