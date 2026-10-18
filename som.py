"""Batch self-organising map, Ward clustering of the map and cluster labelling."""

import heapq
from itertools import combinations

import numpy as np

CHUNK = 16384


def bmn(x: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # best-matching neuron and its distance for every pattern
    w = w.astype(np.float32)
    w2 = (w**2).sum(axis=1)
    idx = np.empty(len(x), dtype=np.int64)
    dist = np.empty(len(x))
    for s in range(0, len(x), CHUNK):
        xc = x[s : s + CHUNK]
        d2 = (xc**2).sum(axis=1)[:, None] - 2.0 * (xc @ w.T) + w2[None]
        best = d2.argmin(axis=1)
        idx[s : s + CHUNK] = best
        dist[s : s + CHUNK] = np.sqrt(np.maximum(d2[np.arange(len(best)), best], 0.0))
    return idx, dist


def topographic_error(x: np.ndarray, w: np.ndarray, adj: np.ndarray) -> float:
    w = w.astype(np.float32)
    w2 = (w**2).sum(axis=1)
    wrong = 0
    for s in range(0, len(x), CHUNK):
        xc = x[s : s + CHUNK]
        d2 = (xc**2).sum(axis=1)[:, None] - 2.0 * (xc @ w.T) + w2[None]
        two = np.argpartition(d2, 1, axis=1)[:, :2]
        wrong += int((~adj[two[:, 0], two[:, 1]]).sum())
    return wrong / len(x)


def train(
    p: dict, x: np.ndarray, weight: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, float]:
    # returns the codebook, the grid adjacency and the final weighted
    # quantisation error
    side = p["side"]
    row, col = np.divmod(np.arange(side * side), side)
    coords = np.stack([col, row], axis=1).astype(np.float64)
    if p["lattice"] == "hex":
        coords[:, 0] += 0.5 * (row % 2)
        coords[:, 1] *= np.sqrt(3) / 2
    grid_d2 = ((coords[:, None] - coords[None]) ** 2).sum(axis=-1)
    adj = np.abs(grid_d2 - 1.0) < 1e-6

    if p["init"] == "sample":
        rng = np.random.default_rng(seed)
        w = x[rng.choice(len(x), len(coords), replace=False)].astype(np.float64)
    else:
        # first principal component along the columns, second along the rows,
        # two standard deviations either side of the mean
        values, vectors = np.linalg.eigh(np.cov(x, rowvar=False))
        lo, hi = coords.min(axis=0), coords.max(axis=0)
        t = (coords - lo) / (hi - lo) * 2 - 1
        std = np.sqrt(values[-1:-3:-1])
        w = x.mean(axis=0) + (t * 2 * std) @ vectors[:, -1:-3:-1].T

    epochs = p["epochs"]
    sigma0 = p["ratio"] * side
    sigma = sigma0 * (p["sigma_t"] / sigma0) ** (np.arange(epochs) / (epochs - 1))
    k = len(w)
    for s in sigma:
        idx, _ = bmn(x, w)
        h = np.exp(-grid_d2 / (2.0 * s**2))
        sums = np.zeros((k, x.shape[1]))
        np.add.at(sums, idx, weight[:, None] * x)
        num = h @ sums
        den = h @ np.bincount(idx, weights=weight, minlength=k)
        updated = den > 0
        w[updated] = num[updated] / den[updated, None]
    _, dist = bmn(x, w)
    return w, adj, float(np.average(dist, weights=weight))


def ward(
    w: np.ndarray, adj: np.ndarray, hits: np.ndarray
) -> tuple[list[tuple[int, int, int]], int]:
    # Ward merges of the occupied neurons, weighted by their hit counts, with
    # only grid-adjacent clusters allowed to merge. Returns the merge sequence
    # (r, s, new id) and the number of merges that had to be made without an
    # adjacent pair, which is one less than the number of islands of occupied
    # neurons.
    alive = set(np.flatnonzero(hits).tolist())
    centre = {k: w[k].astype(np.float64) for k in alive}
    count = {k: float(hits[k]) for k in alive}
    neighbours = {k: set(np.flatnonzero(adj[k]).tolist()) & alive for k in alive}
    heap = []
    for r in alive:
        for s in neighbours[r]:
            if r < s:
                d = count[r] * count[s] / (count[r] + count[s])
                heap.append((d * float(((centre[r] - centre[s]) ** 2).sum()), r, s))
    heapq.heapify(heap)
    merges = []
    unconstrained = 0
    next_id = len(w)
    while len(alive) > 1:
        if not heap:
            unconstrained += len(alive) - 1
            for r, s in combinations(sorted(alive), 2):
                neighbours[r].add(s)
                neighbours[s].add(r)
                d = count[r] * count[s] / (count[r] + count[s])
                heapq.heappush(
                    heap, (d * float(((centre[r] - centre[s]) ** 2).sum()), r, s)
                )
        _, r, s = heapq.heappop(heap)
        if r not in alive or s not in alive:
            continue
        q = next_id
        next_id += 1
        count[q] = count[r] + count[s]
        centre[q] = (count[r] * centre[r] + count[s] * centre[s]) / count[q]
        neighbours[q] = (neighbours.pop(r) | neighbours.pop(s)) - {r, s}
        for t in neighbours[q]:
            neighbours[t] -= {r, s}
            neighbours[t].add(q)
            d = count[q] * count[t] / (count[q] + count[t])
            heapq.heappush(
                heap,
                (d * float(((centre[q] - centre[t]) ** 2).sum()), min(q, t), max(q, t)),
            )
        alive -= {r, s}
        alive.add(q)
        merges.append((r, s, q))
    return merges, unconstrained


def predict(
    merges: list[tuple[int, int, int]],
    hits: np.ndarray,
    c: int,
    winner: np.ndarray,
    y: np.ndarray,
    alpha: float,
    other: np.ndarray,
) -> np.ndarray:
    # cut the merge sequence at c clusters, label each cluster with the class
    # that maximises n_ck / N_k^alpha, and predict for patterns with winners
    # `other`
    member = np.arange(len(hits))
    for r, s, q in merges[: int((hits > 0).sum()) - c]:
        member[(member == r) | (member == s)] = q
    cluster = np.full(len(hits), -1)
    occupied = hits > 0
    cluster[occupied] = np.unique(member[occupied], return_inverse=True)[1]
    counts = np.zeros((cluster.max() + 1, 10))
    np.add.at(counts, (cluster[winner], y), 1)
    prior = np.bincount(y, minlength=10).astype(np.float64)
    label = (counts / prior**alpha).argmax(axis=1)
    return label[cluster[other]]
