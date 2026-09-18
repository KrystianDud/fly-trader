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
        # scratch buffers, so the hot loop allocates nothing
        self._tmp = z()
        self._mask = torch.zeros(self.n, batch, device=self.device, dtype=torch.bool)
        self._spikes = torch.zeros(self.n, batch, device=self.device, dtype=torch.bool)
        self._sf = z()

    def step(self, drive_dt: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt, in place.

        `drive_dt` is the external input already multiplied by dt (see `run`),
        so results stay comparable across integration step sizes. Synaptic input
        arrives as discrete jumps.

        Every update here is in place: at 165,836 neurons times a batch, the
        allocations dominated the sparse matrix multiply.
        """
        p = self.p
        delayed = self.delay_buf[self.buf_pos]

        # g decays, then takes the delayed synaptic input and the external drive
        self.g.mul_(1.0 - p.dt / p.tau).add_(delayed)
        if drive_dt is not None:
            self.g.add_(drive_dt)

        # dv = (v_0 - v + g) * dt / t_mbr, applied only outside the refractory period
        torch.sub(self.g, self.v, out=self._tmp)
        self._tmp.add_(p.v_0).mul_(p.dt / p.t_mbr)
        self._tmp.mul_(torch.le(self.refrac, 0, out=self._mask))
        self.v.add_(self._tmp)

        torch.gt(self.v, p.v_th, out=self._spikes)
        self._spikes.logical_and_(self._mask)

        self.v.masked_fill_(self._spikes, p.v_rst)
        self.g.masked_fill_(self._spikes, 0.0)
        self.refrac.sub_(1).masked_fill_(self._spikes, self.rfc_steps)

        # synaptic current arriving after t_dly
        self._sf.copy_(self._spikes)
        self.delay_buf[self.buf_pos] = torch.mm(self.W, self._sf)
        self.buf_pos = (self.buf_pos + 1) % self.delay_steps
        return self._spikes

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
            drive.mul_(self.p.dt)

        n_rec = self.n if record is None else len(record)
        counts = torch.zeros(n_rec, batch, device=self.device, dtype=self.dtype)
        for _ in range(steps):
            spikes = self.step(drive)
            counts += (spikes if record is None else spikes[record]).to(self.dtype)
        return counts

    def run_binned(
        self,
        ms: float,
        drive_idx: torch.Tensor,
        drive_mV: torch.Tensor,
        record: torch.Tensor,
        bins: int = 4,
    ) -> torch.Tensor:
        """Run and return spike counts per time bin: (bins, n_recorded, batch).

        Timing carries information a single total throws away, and recording it
        costs nothing at extraction time but a re-run to add later.
        """
        batch = drive_mV.shape[1] if drive_mV.dim() == 2 else 1
        self.reset(batch)

        drive = torch.zeros(self.n, batch, device=self.device, dtype=self.dtype)
        d = drive_mV.to(self.device, self.dtype)
        drive[drive_idx] = d if d.dim() == 2 else d.unsqueeze(1).expand(-1, batch)
        drive.mul_(self.p.dt)

        steps = int(round(ms / self.p.dt))
        per_bin = max(1, steps // bins)
        out = torch.zeros(bins, len(record), batch, device=self.device, dtype=self.dtype)

        for s in range(steps):
            spikes = self.step(drive)
            b = min(bins - 1, s // per_bin)
            out[b] += spikes[record].to(self.dtype)
        return out
