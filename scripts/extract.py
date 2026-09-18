"""Run market windows through a brain once, cache the descending activity.

The forward pass does not depend on the label, the horizon or the readout, so
it is computed once per (arm, window) and stored. Every experiment afterwards —
readouts, horizons, cross-validation schemes, learning curves — runs in seconds
on the cached arrays instead of hours on the simulation.

Usage:
    PYTHONPATH=. uv run python scripts/extract.py --arm real --windows 20000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fly_trader import arms, connectome, senses
from fly_trader.lif import FlyBrain, LIFParams

OUT = Path("data/activations")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="real", help="real | rewire0.25 | ... | random")
    ap.add_argument("--windows", type=int, default=0, help="0 = all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--sim-ms", type=float, default=20.0)
    ap.add_argument("--bins", type=int, default=4)
    ap.add_argument("--dt", type=float, default=0.2)
    ap.add_argument("--step", type=int, default=15, help="minutes between decisions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--market", default="data/market/usdjpy_1m.parquet")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.market).stem}_{args.arm}_s{args.seed}_{args.sim_ms:g}ms"

    c = connectome.load(min_weight=5)
    smap = senses.SensoryMap(c)
    W = arms.build(c.W, args.arm, seed=args.seed)
    stats = arms.graph_stats(W)

    bars = pd.read_parquet(args.market)
    feat = senses.features(bars, step=args.step)
    if args.windows:
        feat = feat.iloc[: args.windows]

    dn_ids = c.ids_where(superclass="descending_neuron")
    dn = torch.tensor(c.idx(dn_ids))

    brain = FlyBrain(W, LIFParams(dt=args.dt), device=args.device)
    n = len(feat)
    acts = np.zeros((n, args.bins, len(dn)), dtype=np.uint8)

    print(f"arm={args.arm}  windows={n:,}  neurons={c.n:,}  "
          f"edges={stats['edges']:,}  device={args.device}  batch={args.batch}")
    t0 = time.time()
    for start in range(0, n, args.batch):
        chunk = feat.iloc[start : start + args.batch]
        out = brain.run_binned(
            args.sim_ms, smap.all_idx, smap.drive(chunk), dn, bins=args.bins
        )
        # (bins, neurons, batch) -> (batch, bins, neurons), clipped to uint8
        acts[start : start + len(chunk)] = (
            out.permute(2, 0, 1).clamp(0, 255).to(torch.uint8).cpu().numpy()
        )
        done = start + len(chunk)
        if start % (args.batch * 20) == 0 or done >= n:
            rate = done / max(1e-9, time.time() - t0)
            eta = (n - done) / max(1e-9, rate)
            print(f"  {done:>7,}/{n:,}  {rate:5.2f} win/s  eta {eta / 60:6.1f} min",
                  flush=True)

    np.save(OUT / f"{tag}_acts.npy", acts)
    feat.drop(columns=["close"], errors="ignore").to_parquet(OUT / f"{tag}_feat.parquet")
    meta = {
        "arm": args.arm,
        "seed": args.seed,
        "windows": int(n),
        "bins": int(args.bins),
        "sim_ms": args.sim_ms,
        "dt": args.dt,
        "step_minutes": args.step,
        "market": args.market,
        "from": str(feat.index[0]),
        "to": str(feat.index[-1]),
        "dn_ids": [int(b) for b in dn_ids],
        "dn_types": [str(c.ann.loc[b, "type"]) for b in dn_ids],
        "graph": stats,
        "elapsed_s": round(time.time() - t0, 1),
        "clipped_windows": int((acts == 255).any(axis=(1, 2)).sum()),
    }
    (OUT / f"{tag}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nwrote {OUT / tag}_acts.npy  {acts.nbytes / 1e6:.0f} MB")
    print(f"mean spikes/window {acts.sum(axis=(1, 2)).mean():.0f}, "
          f"active neurons {(acts.sum(axis=(0, 1)) > 0).sum()} of {len(dn_ids)}")
    print(f"elapsed {meta['elapsed_s'] / 60:.1f} min")


if __name__ == "__main__":
    main()
