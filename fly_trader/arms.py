"""Control graphs: what we compare the real connectome against.

The headline experiment is a dose-response curve, not a binary test. Rewiring a
fraction of the edges and watching performance degrade in proportion is a much
stronger structural claim than one scrambled comparison, which can always be
luck.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def degree_preserving_rewire(
    W: sp.csr_matrix, fraction: float = 1.0, seed: int = 0
) -> sp.csr_matrix:
    """Shuffle a fraction of edge targets, preserving every degree exactly.

    Permuting the destination column among a subset of edges leaves each
    neuron's out-degree untouched (sources never move) and its in-degree
    untouched (the multiset of destinations is unchanged). Synaptic weight and
    sign travel with the source, so the weight distribution is also preserved.

    What is destroyed is *which* neuron connects to which: the structure.
    """
    if fraction <= 0:
        return W.copy()

    rng = np.random.default_rng(seed)
    coo = W.tocoo()
    # the matrix is stored as W[post, pre]: row is the target, column the source
    post, pre, val = coo.row.copy(), coo.col.copy(), coo.data.copy()

    n_edges = len(pre)
    k = int(round(fraction * n_edges))
    if k < 2:
        return W.copy()

    pick = rng.choice(n_edges, k, replace=False)
    post[pick] = post[rng.permutation(pick)]

    keep = pre != post  # drop self-loops created by the shuffle
    out = sp.csr_matrix((val[keep], (post[keep], pre[keep])), shape=W.shape)
    out.sum_duplicates()
    return out


def random_graph(W: sp.csr_matrix, seed: int = 0) -> sp.csr_matrix:
    """Erdos-Renyi graph with matched node count, edge count and weight pool.

    Controls for "any large sparse graph would do". Degrees are not preserved,
    so this is a weaker control than the rewire and mostly a sanity floor.
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    coo = W.tocoo()
    n_edges = len(coo.data)

    pre = rng.integers(0, n, n_edges)
    post = rng.integers(0, n, n_edges)
    val = rng.permutation(coo.data)

    keep = pre != post
    out = sp.csr_matrix((val[keep], (post[keep], pre[keep])), shape=W.shape)
    out.sum_duplicates()
    return out


def graph_stats(W: sp.csr_matrix) -> dict:
    """Summary used to verify a control really is matched on what it claims."""
    coo = W.tocoo()
    out_deg = np.bincount(coo.col, minlength=W.shape[0])
    in_deg = np.bincount(coo.row, minlength=W.shape[0])
    return {
        "edges": int(len(coo.data)),
        "synapses": float(np.abs(coo.data).sum()),
        "inhibitory_share": float((coo.data < 0).mean()),
        "out_degree_mean": float(out_deg.mean()),
        "out_degree_sd": float(out_deg.std()),
        "in_degree_mean": float(in_deg.mean()),
        "in_degree_sd": float(in_deg.std()),
        "reciprocity": float((W.multiply(W.T) != 0).nnz / max(1, len(coo.data))),
    }


def build(W: sp.csr_matrix, arm: str, seed: int = 0) -> sp.csr_matrix:
    """Arm names: 'real', 'rewire0.25' ... 'rewire1.0', 'random'."""
    if arm == "real":
        return W
    if arm == "random":
        return random_graph(W, seed)
    if arm.startswith("rewire"):
        return degree_preserving_rewire(W, float(arm[len("rewire"):]), seed)
    raise ValueError(f"unknown arm: {arm}")
