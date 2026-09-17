"""Leaky integrate-and-fire simulation of the connectome, in PyTorch.

Parameters follow Shiu et al. 2024 (Nature 634:210) / philshiu/Drosophila_brain_model:
resting -52 mV, threshold -45 mV, membrane tau 20 ms, synaptic tau 5 ms,
refractory 2.2 ms, synaptic delay 1.8 ms, 0.275 mV per synapse.

The weight matrix is fixed: nothing here is trained.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch


@dataclass
class LIFParams:
    v_0: float = -52.0  # mV, resting
    v_rst: float = -52.0  # mV, reset after spike
    v_th: float = -45.0  # mV, threshold
    t_mbr: float = 20.0  # ms, membrane time constant
    tau: float = 5.0  # ms, synaptic time constant
    t_rfc: float = 2.2  # ms, refractory period
    t_dly: float = 1.8  # ms, synaptic delay
    w_syn: float = 0.275  # mV per synapse
    dt: float = 0.1  # ms, integration step


class FlyBrain:
    """Fixed-weight spiking network. Input is a per-neuron current in mV/ms."""

    def __init__(
        self,
        W: sp.csr_matrix,
        params: LIFParams | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.p = params or LIFParams()
        self.device = torch.device(device)
        self.dtype = dtype
        self.n = W.shape[0]

        coo = W.tocoo()
        idx = torch.from_numpy(np.vstack([coo.row, coo.col])).to(torch.int64)
        val = torch.from_numpy(coo.data.astype(np.float32) * self.p.w_syn)
        self.W = torch.sparse_csr_tensor(
            *self._csr_from_coo(idx, val, self.n), size=(self.n, self.n)
        ).to(self.device, dtype)

        self.delay_steps = max(1, int(round(self.p.t_dly / self.p.dt)))
        self.rfc_steps = int(round(self.p.t_rfc / self.p.dt))
        self.reset()

    @staticmethod
    def _csr_from_coo(idx, val, n):
        order = torch.argsort(idx[0] * n + idx[1])
        row, col, val = idx[0][order], idx[1][order], val[order]
        counts = torch.bincount(row, minlength=n)
        crow = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
        return crow, col, val

    def reset(self, batch: int = 1) -> None:
        """Reset state. `batch` independent flies run at once, sharing one brain."""
        self.batch = batch
        z = lambda: torch.zeros(self.n, batch, device=self.device, dtype=self.dtype)
        self.v = torch.full_like(z(), self.p.v_0)
        self.g = z()
        self.refrac = torch.zeros(self.n, batch, device=self.device, dtype=torch.int32)
        self.delay_buf = [z() for _ in range(self.delay_steps)]
        self.buf_pos = 0

    def step(self, drive: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt.

        `drive` is an external input rate in mV/ms, so results stay comparable
        across integration step sizes. Synaptic input arrives as discrete jumps.
        """
        p = self.p
        delayed = self.delay_buf[self.buf_pos]

        self.g = self.g + (-self.g / p.tau) * p.dt + delayed
        if drive is not None:
            self.g = self.g + drive * p.dt

        active = self.refrac <= 0
        dv = (p.v_0 - self.v + self.g) / p.t_mbr * p.dt
        self.v = torch.where(active, self.v + dv, self.v)

        spikes = (self.v > p.v_th) & active
        self.v = torch.where(spikes, torch.full_like(self.v, p.v_rst), self.v)
        self.g = torch.where(spikes, torch.zeros_like(self.g), self.g)
        self.refrac = torch.where(
            spikes, torch.full_like(self.refrac, self.rfc_steps), self.refrac - 1
        )

        # synaptic current arriving after t_dly
        s = spikes.to(self.dtype)
        self.delay_buf[self.buf_pos] = torch.mm(self.W, s)
        self.buf_pos = (self.buf_pos + 1) % self.delay_steps
        return spikes

    def run(
        self,
        ms: float,
        drive_idx: torch.Tensor | None = None,
        drive_mV: torch.Tensor | None = None,
        record: torch.Tensor | None = None,
        batch: int = 1,
    ) -> torch.Tensor:
        """Run for `ms`, holding a constant drive on drive_idx.

        drive_mV is (n_driven,) or (n_driven, batch): one column per market
        window, so many windows are evaluated in a single pass over the graph.
        Returns spike counts, shape (n_recorded, batch).
        """
        if drive_mV is not None and drive_mV.dim() == 2:
            batch = drive_mV.shape[1]
        self.reset(batch)

        steps = int(round(ms / self.p.dt))
        drive = None
        if drive_idx is not None:
            drive = torch.zeros(self.n, batch, device=self.device, dtype=self.dtype)
            d = drive_mV.to(self.device, self.dtype)
            drive[drive_idx] = d if d.dim() == 2 else d.unsqueeze(1).expand(-1, batch)

        n_rec = self.n if record is None else len(record)
        counts = torch.zeros(n_rec, batch, device=self.device, dtype=self.dtype)
        for _ in range(steps):
            spikes = self.step(drive)
            counts += (spikes if record is None else spikes[record]).to(self.dtype)
        return counts
