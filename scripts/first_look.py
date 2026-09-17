"""Go/no-go: does the fixed brain respond differently to different markets?

If descending-neuron activity is flat across wildly different market states,
the approach is dead and no readout can save it.
"""

import time

import numpy as np
import pandas as pd
import torch

from fly_trader import connectome, senses
from fly_trader.lif import FlyBrain, LIFParams

N_WINDOWS = 256
SIM_MS = 20.0

torch.set_num_threads(8)
rng = np.random.default_rng(0)

print("loading brain...")
c = connectome.load(min_weight=5)
smap = senses.SensoryMap(c)
print(f"  {c.n:,} neurons, {c.W.nnz:,} connections")
print(f"  input channels: {smap.summary()}")

print("loading market...")
bars = pd.read_parquet("data/market/usdjpy_1m.parquet")
feat = senses.features(bars)
print(f"  {len(feat):,} decision windows, {feat.index.min()} to {feat.index.max()}")

# sample windows spread across the whole period, not one quiet week
sample = feat.iloc[np.sort(rng.choice(len(feat), N_WINDOWS, replace=False))]
drive = smap.drive(sample)

dn_ids = c.ids_where(superclass="descending_neuron")
dn = torch.tensor(c.idx(dn_ids))
gf = torch.tensor(c.idx(c.ids_where(type="DNp01")))  # Giant Fiber

print(f"running {N_WINDOWS} windows x {SIM_MS} ms of brain time...")
brain = FlyBrain(c.W, LIFParams(dt=0.2))
t0 = time.time()
counts = brain.run(SIM_MS, smap.all_idx, drive, record=dn).numpy().T  # (windows, DNs)
gf_counts = brain.run(SIM_MS, smap.all_idx, drive, record=gf).numpy().T
print(f"  {time.time() - t0:.0f}s total, {(time.time() - t0) / N_WINDOWS:.2f}s per window\n")

total = counts.sum(axis=1)
active = (counts > 0).sum(axis=1)
print("=== does the brain respond at all? ===")
print(f"descending spikes per window: mean {total.mean():.0f}, "
      f"min {total.min():.0f}, max {total.max():.0f}, sd {total.std():.0f}")
print(f"active descending neurons:    mean {active.mean():.0f} of {len(dn_ids)}, "
      f"min {active.min()}, max {active.max()}")
dead = (counts.sum(axis=0) == 0).sum()
print(f"descending neurons never firing: {dead} of {len(dn_ids)}")

print("\n=== does the response track the market? ===")
channels = ["fast_up", "fast_down", "slow_up", "slow_down", "volatility", "looming"]
for ch in channels:
    x = sample[ch].to_numpy()
    if x.std() == 0:
        print(f"  {ch:11s}: channel silent in this sample")
        continue
    r = np.corrcoef(x, total)[0, 1]
    # strongest individual descending neuron for this channel
    with np.errstate(invalid="ignore"):
        rs = np.array([np.corrcoef(x, counts[:, i])[0, 1] for i in range(counts.shape[1])])
    rs = np.nan_to_num(rs)
    best = np.abs(rs).argmax()
    print(f"  {ch:11s}: r(total spikes) = {r:+.2f}   "
          f"best single DN {c.ann.loc[dn_ids[best], 'type']}: r = {rs[best]:+.2f}")

print("\n=== is the escape circuit market-aware? ===")
loom = sample["looming"].to_numpy()
gf_total = gf_counts.sum(axis=1)
if gf_total.std() > 0:
    print(f"  Giant Fiber (DNp01) spikes vs looming signal: "
          f"r = {np.corrcoef(loom, gf_total)[0, 1]:+.2f}")
    q = pd.qcut(loom, 4, labels=False, duplicates="drop")
    for i in range(q.max() + 1):
        print(f"    looming quartile {i + 1}: {gf_total[q == i].mean():.1f} GF spikes")
else:
    print(f"  Giant Fiber silent in all windows (total {gf_total.sum():.0f})")

print("\n=== is the pattern distinguishable across states? ===")
# how much of the variation in descending activity is explained by market state?
from sklearn.decomposition import PCA

p = PCA(n_components=5).fit(counts)
print(f"  top-5 PCs explain {p.explained_variance_ratio_.sum():.0%} of DN variance")
pcs = p.transform(counts)
for i in range(3):
    best_ch = max(channels, key=lambda ch: abs(np.corrcoef(sample[ch], pcs[:, i])[0, 1])
                  if sample[ch].std() > 0 else 0)
    r = np.corrcoef(sample[best_ch], pcs[:, i])[0, 1]
    print(f"  PC{i + 1} ({p.explained_variance_ratio_[i]:.0%}): "
          f"best matched by {best_ch}, r = {r:+.2f}")
