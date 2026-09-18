"""The connectome as a trainable architecture, not a frozen filter.

Everything else in this project freezes the wiring and trains only a readout,
so that any advantage is attributable to biology. That design answers one
question and refuses another: what happens if you *train* over the graph, the
way FlyGM trained per-neuron parameters and beat non-connectome baselines on
locomotion.

This is that experiment. The connectome supplies the topology and the sign of
every connection — which neuron talks to which, and whether it excites or
inhibits — and gradient descent supplies everything else: how strongly each
neuron responds, how signals combine, how the input enters and the decision
comes out.

The attribution is therefore weaker by design, and the controls carry the
weight instead. The same architecture is trained on a degree-matched scramble
and on a random graph. If the real topology is doing work, it shows up as a gap
between arms trained identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


def normalise(W: sp.csr_matrix) -> sp.csr_matrix:
    """Scale each neuron's incoming weights so message passing stays stable.

    Without this, hub neurons receiving thousands of connections saturate and
    the rest of the graph never gets a say.
    """
    deg = np.abs(W).sum(axis=1).A.ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ W


@dataclass
class Config:
    channels: int = 8        # latent state per neuron
    steps: int = 4           # message-passing rounds
    hidden: int = 32         # width of the shared update network
    readout_hidden: int = 64


class ConnectomeNet(nn.Module):
    """Message passing along real synapses, with trained neuron properties."""

    def __init__(
        self,
        W: sp.csr_matrix,
        in_idx: torch.Tensor,
        out_idx: torch.Tensor,
        n_features: int,
        cfg: Config | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.cfg = cfg or Config()
        self.n = W.shape[0]
        self.device = torch.device(device)

        A = normalise(W).tocsr()
        self.A = torch.sparse_csr_tensor(
            torch.from_numpy(A.indptr.astype(np.int64)),
            torch.from_numpy(A.indices.astype(np.int64)),
            torch.from_numpy(A.data.astype(np.float32)),
            size=A.shape,
        ).to(self.device)

        self.in_idx = in_idx.to(self.device)
        self.out_idx = out_idx.to(self.device)
        c = self.cfg.channels

        # how the market enters the brain: features -> activity on sensory cells
        self.encoder = nn.Linear(n_features, len(in_idx) * c)

        # per-neuron properties: the biological free parameters
        self.gain = nn.Parameter(torch.ones(self.n, c))
        self.bias = nn.Parameter(torch.zeros(self.n, c))

        # one update rule shared by every neuron, as in the animal
        self.update = nn.Sequential(
            nn.Linear(2 * c, self.cfg.hidden), nn.GELU(), nn.Linear(self.cfg.hidden, c)
        )

        # the decision, read from the descending neurons only
        self.decoder = nn.Sequential(
            nn.Linear(len(out_idx) * c, self.cfg.readout_hidden),
            nn.GELU(),
            nn.Linear(self.cfg.readout_hidden, 1),
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, c = x.shape[0], self.cfg.channels
        h = torch.zeros(self.n, B, c, device=self.device)
        h[self.in_idx] = self.encoder(x).view(B, len(self.in_idx), c).permute(1, 0, 2)

        for _ in range(self.cfg.steps):
            # one hop along the connectome: (n, B*c) keeps the sparse op 2-D
            msg = torch.mm(self.A, h.reshape(self.n, B * c)).view(self.n, B, c)
            h = h + self.update(
                torch.cat([h * self.gain.unsqueeze(1), msg], dim=-1)
            ) + self.bias.unsqueeze(1)
            h = torch.tanh(h)  # neurons saturate; so does this

        out = h[self.out_idx].permute(1, 0, 2).reshape(B, -1)
        return self.decoder(out).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class MatchedMLP(nn.Module):
    """A conventional network with a comparable parameter budget.

    The control for "you just need a model this big", as opposed to "you need
    this particular graph".
    """

    def __init__(self, n_features: int, target_params: int):
        super().__init__()
        # solve roughly for a width giving the same parameter count
        width = max(32, int((target_params / 3) ** 0.5))
        self.net = nn.Sequential(
            nn.Linear(n_features, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
