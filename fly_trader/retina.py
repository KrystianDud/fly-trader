"""Show the price chart to the fly's eye.

Everywhere else in this project the market arrives as scalars: momentum,
volatility, a crash signal. That leaves 89,403 optic-lobe neurons — the largest
structure in the brain — doing nothing.

Here the chart is drawn as an image on the retinotopic lattice and played as a
short movie across the eye during the simulation, so the price line *moves*.
That matters: T4 and T5 are direction-selective motion detectors, and they only
respond to something moving. A static image would engage edge detection alone
and waste half the machinery.

The claim being tested is narrow and falsifiable: if chart shape carries
information, a visual system evolved to extract shape should find more of it
than a degree-matched scramble of the same wiring. No pattern is defined,
labelled or hard-coded anywhere here — the fly gets pixels.

Entry point is the lamina, not the photoreceptors, because the retina is not in
the imaged volume:
  L1 -> ON pathway  (brightness increase)
  L2 -> OFF pathway (brightness decrease)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .connectome import Connectome

ON_TYPE, OFF_TYPE = "L1", "L2"
MAX_DRIVE = 20.0  # mV/ms at full contrast, matching senses.py


@dataclass
class Retina:
    """Maps image pixels onto lamina neurons via their column coordinates."""

    conn: Connectome
    side: str = "R"          # one eye; the other is a mirror
    frames: int = 8          # movie frames per decision
    bars: int = 64           # chart width in 1-minute bars

    def __post_init__(self):
        ann = self.conn.ann
        cols = []
        for t in (ON_TYPE, OFF_TYPE):
            s = ann[(ann.type == t) & ann.assignedOlHex1.notna()]
            if self.side:
                s = s[s.rootSide.fillna(s.somaSide) == self.side]
            cols.append(s)
        self.on_cells, self.off_cells = cols

        # lattice extent from the cells we actually have
        h1 = pd.concat([self.on_cells.assignedOlHex1, self.off_cells.assignedOlHex1])
        h2 = pd.concat([self.on_cells.assignedOlHex2, self.off_cells.assignedOlHex2])
        self.h1_min, self.h1_max = int(h1.min()), int(h1.max())
        self.h2_min, self.h2_max = int(h2.min()), int(h2.max())
        self.width = self.h1_max - self.h1_min + 1
        self.height = self.h2_max - self.h2_min + 1

        self.on_idx, self.on_px = self._index(self.on_cells)
        self.off_idx, self.off_px = self._index(self.off_cells)

    def _index(self, cells: pd.DataFrame):
        """Neuron indices and the pixel each one looks at."""
        ids = cells.index.to_numpy()
        keep = np.array([b in self.conn.index for b in ids])
        cells = cells[keep]
        idx = np.array([self.conn.index[int(b)] for b in cells.index], dtype=np.int64)
        x = cells.assignedOlHex1.to_numpy(dtype=int) - self.h1_min
        y = cells.assignedOlHex2.to_numpy(dtype=int) - self.h2_min
        return torch.from_numpy(idx), (y, x)

    @property
    def all_idx(self) -> torch.Tensor:
        return torch.cat([self.on_idx, self.off_idx])

    def summary(self) -> str:
        return (
            f"{self.width}x{self.height} columns, "
            f"ON({ON_TYPE})={len(self.on_idx)} OFF({OFF_TYPE})={len(self.off_idx)}, "
            f"{self.frames} frames per decision"
        )

    # ------------------------------------------------------------ rendering

    def render(self, closes: np.ndarray) -> np.ndarray:
        """Chart window -> movie of shape (frames, height, width), values 0..1.

        Each frame advances the window by one bar, so the line drifts leftwards
        across the eye and the motion detectors have something to detect.
        """
        need = self.bars + self.frames
        if len(closes) < need:
            closes = np.concatenate([np.full(need - len(closes), closes[0]), closes])
        closes = closes[-need:]

        movie = np.zeros((self.frames, self.height, self.width), dtype=np.float32)
        for f in range(self.frames):
            w = closes[f : f + self.bars]
            lo, hi = w.min(), w.max()
            span = hi - lo
            if span <= 0:
                movie[f, self.height // 2, :] = 1.0
                continue
            # price -> row, time -> column
            cols = np.linspace(0, self.width - 1, self.bars).astype(int)
            rows = ((w - lo) / span * (self.height - 1)).astype(int)
            for c, r in zip(cols, rows):
                movie[f, r, c] = 1.0
            # thicken the line so a single-pixel trace is visible to a coarse eye
            movie[f] = np.maximum(movie[f], np.roll(movie[f], 1, axis=0))
        return movie

    def drive(self, movie: np.ndarray) -> torch.Tensor:
        """Movie -> (frames, n_input_neurons) drive in mV/ms.

        ON cells take brightness, OFF cells take its complement, which is how
        the lamina splits the signal in the animal.
        """
        out = torch.zeros(len(movie), len(self.on_idx) + len(self.off_idx))
        for f, img in enumerate(movie):
            on = img[self.on_px]
            off = 1.0 - img[self.off_px]
            out[f] = torch.from_numpy(np.concatenate([on, off])) * MAX_DRIVE
        return out

    def batch_drive(self, bars: pd.DataFrame, ends: np.ndarray) -> torch.Tensor:
        """Drive for many decisions: (frames, n_input_neurons, batch)."""
        closes = bars["close"].to_numpy()
        per = [
            self.drive(self.render(closes[: e + 1]))
            for e in ends
        ]
        return torch.stack(per, dim=2)
