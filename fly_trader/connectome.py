"""Load the MaleCNS connectome into a signed sparse adjacency matrix.

Data: https://male-cns.janelia.org/download/ (CC-BY 4.0), v1.0, minconf 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / "data" / "connectome"

# Neurotransmitter -> synaptic sign. Acetylcholine is the main excitatory
# transmitter in the fly; GABA and glutamate are inhibitory (glutamate acts on
# the inhibitory GluCl receptor in Drosophila). Modulators are left excitatory
# but weak; they are a small minority of neurons.
NT_SIGN = {
    "acetylcholine": +1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "dopamine": +1.0,
    "serotonin": +1.0,
    "octopamine": +1.0,
    "histamine": -1.0,
    "unclear": +1.0,
}


@dataclass
class Connectome:
    body_ids: np.ndarray  # (N,) bodyId per index
    index: dict[int, int]  # bodyId -> index
    W: sp.csr_matrix  # (N, N) signed weights, W[post, pre]
    ann: pd.DataFrame  # annotations indexed by bodyId

    @property
    def n(self) -> int:
        return len(self.body_ids)

    def ids_where(self, **kwargs) -> np.ndarray:
        """bodyIds whose annotation columns match the given values.

        Values may be a scalar, a list, or a callable taking the column.
        """
        mask = pd.Series(True, index=self.ann.index)
        for col, want in kwargs.items():
            column = self.ann[col]
            if callable(want):
                mask &= want(column)
            elif isinstance(want, (list, tuple, set)):
                mask &= column.isin(want)
            else:
                mask &= column == want
        return self.ann.index[mask].to_numpy()

    def idx(self, body_ids) -> np.ndarray:
        return np.array([self.index[b] for b in body_ids if b in self.index], dtype=np.int64)


def load(min_weight: int = 5, signed: bool = True) -> Connectome:
    """Build the neuron-to-neuron graph, keeping edges of at least min_weight synapses."""
    ann = pd.read_feather(DATA / "annotations.feather")
    ann = ann[ann.superclass.notna()].set_index("bodyId")

    nt = pd.read_feather(
        DATA / "neurotransmitters.feather", columns=["body", "consensus_nt"]
    ).drop_duplicates("body").set_index("body")["consensus_nt"]

    w = pd.read_feather(DATA / "weights.feather")
    w = w[w.weight >= min_weight]
    keep = np.isin(w.body_pre, ann.index) & np.isin(w.body_post, ann.index)
    w = w[keep]

    body_ids = np.union1d(w.body_pre.unique(), w.body_post.unique())
    index = {int(b): i for i, b in enumerate(body_ids)}

    pre = w.body_pre.map(index).to_numpy()
    post = w.body_post.map(index).to_numpy()
    val = w.weight.to_numpy(dtype=np.float32)

    if signed:
        sign = (
            pd.Series(body_ids)
            .map(nt)
            .map(NT_SIGN)
            .fillna(1.0)
            .to_numpy(dtype=np.float32)
        )
        val = val * sign[pre]

    W = sp.csr_matrix((val, (post, pre)), shape=(len(body_ids), len(body_ids)))
    ann = ann.loc[ann.index.isin(body_ids)]
    return Connectome(body_ids=body_ids, index=index, W=W, ann=ann)
