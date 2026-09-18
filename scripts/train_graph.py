"""Train over the connectome graph and compare against its own scramble.

Same purged walk-forward protocol, same costs, same labels as the frozen-weight
experiment — only the model changes. The question is no longer "does biology
help a fixed filter" but "does biological topology help a trained one".

Usage:
    PYTHONPATH=. uv run python scripts/train_graph.py --arms real rewire1.0 mlp
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from fly_trader import arms, connectome, graphmodel, readout, senses, validation

REPORTS = Path("reports")
PERIODS_PER_YEAR = 252 * 24 * 4
COST = 0.00006
FEATURES = ["fast_up", "fast_down", "slow_up", "slow_down", "volatility", "looming"]


def train_fold(model, Xtr, ytr, Xte, epochs, batch, lr, device, log=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=max(1, epochs * (len(Xtr) // batch + 1))
    )
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    ytr_t = ytr_t / ytr_t.std().clamp_min(1e-12)

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr_t), device=device)
        total = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i : i + batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(Xtr_t[b]), ytr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss) * len(b)
        if log and (ep + 1) % max(1, epochs // 3) == 0:
            print(f"      epoch {ep + 1}/{epochs} loss {total / len(perm):.4f}", flush=True)

    model.eval()
    out = []
    with torch.no_grad():
        Xte_t = torch.tensor(Xte, device=device)
        for i in range(0, len(Xte_t), batch):
            out.append(model(Xte_t[i : i + batch]).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["real", "rewire1.0", "random", "mlp"])
    ap.add_argument("--windows", type=int, default=30000)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--channels", type=int, default=8)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    c = connectome.load(min_weight=5)
    smap = senses.SensoryMap(c)
    bars = pd.read_parquet("data/market/usdjpy_1m.parquet")
    feat = senses.features(bars).iloc[: args.windows]

    X = feat[FEATURES].to_numpy(dtype=np.float32)
    y = feat["ret_fwd"].to_numpy(dtype=np.float64)
    n = len(feat)

    folds = validation.purged_walk_forward(n, n_folds=args.folds, horizon=1, embargo=0.01)
    validation.assert_no_leakage(folds, horizon=1)

    dn = torch.tensor(c.idx(c.ids_where(superclass="descending_neuron")))
    cfg = graphmodel.Config(channels=args.channels, steps=args.steps)
    print(f"{n:,} decisions, {len(folds)} folds, device {args.device}\n")

    rows = []
    for arm in args.arms:
        t0 = time.time()
        preds = np.full(n, np.nan)

        for k, f in enumerate(folds):
            if arm == "mlp":
                probe = graphmodel.ConnectomeNet(
                    c.W, smap.all_idx, dn, len(FEATURES), cfg, args.device
                )
                target = probe.n_params()
                del probe
                model = graphmodel.MatchedMLP(len(FEATURES), target).to(args.device)
            else:
                W = arms.build(c.W, arm, seed=args.seed)
                model = graphmodel.ConnectomeNet(
                    W, smap.all_idx, dn, len(FEATURES), cfg, args.device
                )
            if k == 0:
                print(f"  {arm}: {model.n_params():,} trainable parameters", flush=True)

            preds[f.test] = train_fold(
                model, X[f.train], y[f.train], X[f.test],
                args.epochs, args.batch, args.lr, args.device, log=(k == 0),
            )
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

        m = ~np.isnan(preds)
        ic = float(stats.spearmanr(preds[m], y[m]).statistic)

        # same trading rule as everywhere else: gate, then cost every change
        gate = np.quantile(np.abs(preds[m]), 0.6)
        pos = np.where(np.abs(preds[m]) < gate, 0.0, np.sign(preds[m]))
        turn = np.abs(np.diff(np.concatenate([[0.0], pos])))
        net = pos * y[m] - turn * COST
        sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR))
        hit = float(((pos * y[m]) > 0)[pos != 0].mean())

        rows.append({"arm": arm, "ic": ic, "hit_rate": hit, "sharpe": sharpe,
                     "net_mean_bp": float(net.mean() * 1e4),
                     "minutes": round((time.time() - t0) / 60, 1)})
        print(f"  {arm:12s} IC {ic:+.4f}  hit {hit:.3f}  net {net.mean() * 1e4:+.3f} bp  "
              f"Sharpe {sharpe:+.2f}  ({(time.time() - t0) / 60:.1f} min)\n", flush=True)

    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (REPORTS / f"trained_{stamp}.json").write_text(
        json.dumps({"config": vars(args), "arms": rows}, indent=2, default=float)
    )

    real = next((r for r in rows if r["arm"] == "real"), None)
    rew = next((r for r in rows if r["arm"] == "rewire1.0"), None)
    if real and rew:
        print(f"topology advantage: IC {real['ic'] - rew['ic']:+.4f} "
              f"(real {real['ic']:+.4f} vs rewired {rew['ic']:+.4f})")
    print(f"report: reports/trained_{stamp}.json")


if __name__ == "__main__":
    main()
