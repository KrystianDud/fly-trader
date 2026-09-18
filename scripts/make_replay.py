"""Simulate a contiguous stretch of market and write a replay file for the 3D room.

The decision rule here is PROVISIONAL: there is no trained readout yet, so
direction comes from descending neurons that happened to track momentum, and
sells come from the Giant Fiber escape response. Labelled as such in the UI.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fly_trader import connectome, senses
from fly_trader.lif import FlyBrain, LIFParams

N = 300  # consecutive 15-minute decisions (~3 trading days)
SIM_MS = 20.0
PIP = 0.01  # USD/JPY
COST_PIPS = 0.8  # spread + slippage assumption

torch.set_num_threads(8)

c = connectome.load(min_weight=5)
smap = senses.SensoryMap(c)
bars = pd.read_parquet("data/market/usdjpy_1m.parquet")
feat = senses.features(bars)

# pick the most eventful stretch: highest summed absolute move over N windows
move = feat["ret_fwd"].abs().rolling(N).sum()
end = int(np.nanargmax(move.to_numpy())) + 1
window = feat.iloc[max(0, end - N) : end]
print(f"replay: {window.index[0]} -> {window.index[-1]}  ({len(window)} decisions)")

dn_ids = c.ids_where(superclass="descending_neuron")
dn = torch.tensor(c.idx(dn_ids))
gf_ids = c.ids_where(type="DNp01")

brain = FlyBrain(c.W, LIFParams(dt=0.2))
counts = brain.run(SIM_MS, smap.all_idx, smap.drive(window), record=dn).numpy().T

# keep only neurons that ever fire, for a compact brain display
live = counts.sum(axis=0) > 0
counts_live = counts[:, live].astype(int)
live_types = [str(c.ann.loc[b, "type"]) for b in dn_ids[live]]
gf_col = [i for i, b in enumerate(dn_ids[live]) if b in set(gf_ids)]
gf = counts_live[:, gf_col].sum(axis=1) if gf_col else np.zeros(len(window))

# provisional decision rule
def pop(types):
    cols = [i for i, t in enumerate(live_types) if t in types]
    return counts_live[:, cols].sum(axis=1) if cols else np.zeros(len(window))

up = pop({"DNb05", "DNa02"})
down = pop({"DNa10", "DNa07"})
drive_net = (up - down).astype(float)
scale = np.abs(drive_net).max() or 1.0
score = drive_net / scale
escape = gf > np.quantile(gf, 0.8)

action = np.where(escape, "SELL", np.where(score > 0.15, "BUY",
                  np.where(score < -0.15, "SELL", "FLAT")))
confidence = np.clip(np.abs(score) + 0.4 * escape, 0, 1)

pos = np.where(action == "BUY", 1, np.where(action == "SELL", -1, 0))
ret_pips = window["ret_fwd"].to_numpy() * window["close"].to_numpy() / PIP
pnl = pos * ret_pips - (np.abs(np.diff(np.concatenate([[0], pos]))) > 0) * COST_PIPS

frames = []
for i, (ts, row) in enumerate(window.iterrows()):
    frames.append({
        "t": ts.isoformat(),
        "o": round(float(row["close"]), 3),
        "c": round(float(row["close"]), 3),
        "action": str(action[i]),
        "conf": round(float(confidence[i]), 3),
        "score": round(float(score[i]), 3),
        "gf": int(gf[i]),
        "spikes": int(counts_live[i].sum()),
        "pnl": round(float(pnl[i]), 2),
        "signals": {k: round(float(row[k]), 3)
                    for k in ("fast_up", "fast_down", "slow_up", "slow_down",
                              "volatility", "looming")},
        "brain": counts_live[i].tolist(),
    })

# 1-minute price context for the monitor
lo, hi = window.index[0], window.index[-1]
minute = bars.loc[lo:hi, "close"]
minute = minute.iloc[:: max(1, len(minute) // 1500)]

out = {
    "meta": {
        "instrument": "USD/JPY",
        "from": str(window.index[0]),
        "to": str(window.index[-1]),
        "decisions": len(frames),
        "step_minutes": 15,
        "neurons": int(c.n),
        "connections": int(c.W.nnz),
        "descending_shown": int(live.sum()),
        "cost_pips": COST_PIPS,
        "provisional": True,
    },
    "dn_types": live_types,
    "gf_index": gf_col,
    "frames": frames,
    "minute": [{"t": t.isoformat(), "c": round(float(v), 3)} for t, v in minute.items()],
}

Path("data/replay.json").write_text(json.dumps(out))
print(f"wrote data/replay.json  ({Path('data/replay.json').stat().st_size / 1e6:.1f} MB)")
print(f"actions: {pd.Series(action).value_counts().to_dict()}")
print(f"total pnl: {pnl.sum():.1f} pips over {len(frames)} decisions")
