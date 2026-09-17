"""Turn market state into activity in named fly sensory circuits.

Every channel targets a real, identified cell type. Nothing here is learned:
this is a fixed wiring of market signals onto sense organs, chosen once and
then left alone, so that what the readout learns is only how to interpret the
brain's response.

Channel                     Cell types            Why
-----------------------------------------------------------------------------
fast momentum up/down       T4c/T5c, T4d/T5d      vertical motion detectors,
                                                  ON and OFF pathways
slow momentum up/down       T4a/T5a, T4b/T5b      horizontal motion detectors,
                                                  a separate population that
                                                  converges downstream
volatility                  tactile mechanosensory wind/turbulence on the body
crash / spike               LC4, LPLC2            looming detectors that drive
                                                  the Giant Fiber escape
bad news (Jev)              ORN_DA2               aversive odour (geosmin)
good news (Jev)             ORN_DM1               attractive odour (vinegar)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

from .connectome import Connectome

CHANNELS: dict[str, dict] = {
    "fast_up": {"types": ["T4c", "T5c"]},
    "fast_down": {"types": ["T4d", "T5d"]},
    "slow_up": {"types": ["T4a", "T5a"]},
    "slow_down": {"types": ["T4b", "T5b"]},
    "volatility": {"class": "mechanosensory_tactile", "limit": 600},
    "looming": {"types": ["LC4", "LPLC2"]},
    "news_bad": {"types": ["ORN_DA2"]},
    "news_good": {"types": ["ORN_DM1"]},
}

# Peak input rate (mV/ms) for a channel at full strength.
MAX_DRIVE = 20.0


@dataclass
class SensoryMap:
    conn: Connectome
    idx: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        for name, spec in CHANNELS.items():
            if "types" in spec:
                ids = self.conn.ids_where(type=spec["types"])
            else:
                ids = self.conn.ids_where(**{"class": spec["class"]})
            j = self.conn.idx(ids)
            if "limit" in spec and len(j) > spec["limit"]:
                rng = np.random.default_rng(0)
                j = np.sort(rng.choice(j, spec["limit"], replace=False))
            self.idx[name] = torch.from_numpy(j)

    @property
    def all_idx(self) -> torch.Tensor:
        return torch.cat([self.idx[n] for n in CHANNELS])

    def drive(self, features: pd.DataFrame) -> torch.Tensor:
        """Build a (n_input_neurons, n_windows) drive matrix from features.

        Features are expected in [0, 1]; missing columns stay silent.
        """
        cols = []
        for name in CHANNELS:
            n = len(self.idx[name])
            if name in features:
                v = torch.from_numpy(
                    features[name].to_numpy(dtype=np.float32)
                ).clamp(0, 1)
            else:
                v = torch.zeros(len(features), dtype=torch.float32)
            cols.append(v.unsqueeze(0).expand(n, -1) * MAX_DRIVE)
        return torch.cat(cols, dim=0).contiguous()

    def summary(self) -> str:
        return ", ".join(f"{n}={len(j)}" for n, j in self.idx.items())


def features(
    bars: pd.DataFrame,
    step: int = 15,
    fast: int = 5,
    slow: int = 60,
    vol_window: int = 60,
) -> pd.DataFrame:
    """Market features on a decision grid, each squashed into [0, 1].

    bars: 1-minute OHLC indexed by timestamp.
    step: minutes between decisions.
    """
    close = bars["close"]
    ret1 = np.log(close).diff()
    vol = ret1.rolling(vol_window).std()

    fast_mom = (np.log(close) - np.log(close.shift(fast))) / (vol * np.sqrt(fast))
    slow_mom = (np.log(close) - np.log(close.shift(slow))) / (vol * np.sqrt(slow))

    # volatility relative to its own recent normal
    vol_rel = vol / vol.rolling(vol_window * 8).median() - 1.0

    # worst adverse excursion in the last `fast` minutes, in units of vol
    drop = (bars["low"].rolling(fast).min() / close.shift(fast) - 1.0) / vol

    out = pd.DataFrame(index=bars.index)
    squash = lambda x, k: (x.abs() / k).clip(0, 1)
    out["fast_up"] = squash(fast_mom.clip(lower=0), 2.0)
    out["fast_down"] = squash(fast_mom.clip(upper=0), 2.0)
    out["slow_up"] = squash(slow_mom.clip(lower=0), 2.0)
    out["slow_down"] = squash(slow_mom.clip(upper=0), 2.0)
    out["volatility"] = squash(vol_rel.clip(lower=0), 1.5)
    out["looming"] = squash(drop.clip(upper=0), 3.0)
    out["ret_fwd"] = np.log(close.shift(-step)) - np.log(close)  # label
    out["close"] = close

    return out.iloc[::step].dropna()
