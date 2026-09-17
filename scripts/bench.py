"""Build the graph and time the simulation on this machine."""

import time

import torch

from fly_trader import connectome
from fly_trader.lif import FlyBrain

t0 = time.time()
c = connectome.load(min_weight=5)
print(f"graph: {c.n:,} neurons, {c.W.nnz:,} edges  ({time.time() - t0:.1f}s to load)")

dn = c.idx(c.ids_where(superclass="descending_neuron"))
vis = c.idx(c.ids_where(superclass="visual_projection"))
print(f"descending neurons: {len(dn):,}   visual projection: {len(vis):,}")

for device in ["cpu", "mps"]:
    if device == "mps" and not torch.backends.mps.is_available():
        continue
    brain = FlyBrain(c.W, device=device)
    drive_idx = torch.tensor(vis[:200])
    drive = torch.full((200,), 0.5)

    brain.run(1.0, drive_idx, drive)  # warm-up
    brain.reset()
    t0 = time.time()
    counts = brain.run(20.0, drive_idx, drive, record=torch.tensor(dn))
    dt = time.time() - t0
    print(
        f"{device}: 20 ms of brain time in {dt:.2f}s "
        f"({dt / 20:.3f}s per simulated ms), "
        f"{int(counts.sum())} descending spikes"
    )
