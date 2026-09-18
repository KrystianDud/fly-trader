"""Compare arms on cached activations and write a run report.

Nothing here touches the simulation: it reads the cached descending activity,
so the whole comparison runs in seconds and can be repeated freely.
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fly_trader import readout, validation

ACTS = Path("data/activations")
REPORTS = Path("reports")
PERIODS_PER_YEAR = 252 * 24 * 4  # 15-minute bars, 24h FX market


def load(tag: str):
    acts = np.load(ACTS / f"{tag}_acts.npy")
    feat = pd.read_parquet(ACTS / f"{tag}_feat.parquet")
    meta = json.loads((ACTS / f"{tag}_meta.json").read_text())
    return acts, feat, meta


def run_arm(tag: str, folds, weights, alpha: float, bins: str, sizes: list[int]):
    acts, feat, meta = load(tag)
    X = readout.design_matrix(acts, bins=bins)
    y = feat["ret_fwd"].to_numpy(dtype=np.float64)

    res = readout.fit(X, y, folds, alpha=alpha, weights=weights)
    metrics = readout.evaluate(res, y, cost=COST, periods_per_year=PERIODS_PER_YEAR)

    # learning curve: where an inductive-bias advantage should live
    curve = []
    for size in sizes:
        small = [
            validation.Fold(train=f.train[-size:], test=f.test)
            for f in folds if len(f.train) >= size
        ]
        if not small:
            continue
        r = readout.fit(X, y, small, alpha=alpha, weights=weights)
        m = readout.evaluate(r, y, cost=COST, periods_per_year=PERIODS_PER_YEAR)
        curve.append({"train_size": size, "ic": m["ic"], "sharpe": m["sharpe"]})

    metrics["learning_curve"] = curve
    metrics["arm"] = meta["arm"]
    metrics["reciprocity"] = meta["graph"]["reciprocity"]
    metrics["windows"] = meta["windows"]
    return metrics


COST = 0.00006  # ~0.8 pip on USD/JPY around 150, as a fraction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="usdjpy_1m")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sim-ms", default="20")
    ap.add_argument("--alpha", type=float, default=1e4)
    ap.add_argument("--bins", default="all", choices=["all", "sum"])
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    tags = sorted(
        p.name[: -len("_meta.json")] for p in ACTS.glob(f"{args.market}_*_meta.json")
    )
    tags = [t for t in tags if t.endswith(f"s{args.seed}_{args.sim_ms}ms")]
    if not tags:
        raise SystemExit(f"no cached activations for {args.market}")

    _, feat0, meta0 = load(tags[0])
    n = meta0["windows"]
    folds = validation.purged_walk_forward(
        n, n_folds=args.folds, horizon=args.horizon, embargo=0.01
    )
    validation.assert_no_leakage(folds, horizon=args.horizon)
    weights = validation.sample_weights(n, horizon=args.horizon)

    sizes = [s for s in (500, 1000, 2500, 5000, 10000, 20000) if s <= n]

    rows = []
    for tag in tags:
        m = run_arm(tag, folds, weights, args.alpha, args.bins, sizes)
        rows.append(m)
        print(f"{m['arm']:12s} IC {m['ic']:+.4f}  hit {m['hit_rate']:.3f}  "
              f"net {m['net_mean_bp']:+.3f} bp  Sharpe {m['sharpe']:+.2f}  "
              f"trades {m['trade_share']:.2f}", flush=True)

    # deflate the best arm by how many configurations were tried
    trial_sharpes = np.array([r["sharpe"] for r in rows])
    best = max(rows, key=lambda r: r["sharpe"])
    dsr = validation.deflated_sharpe(
        best["sharpe"], best["net_returns"], trial_sharpes, PERIODS_PER_YEAR
    )

    real = next((r for r in rows if r["arm"] == "real"), None)
    rewired = next((r for r in rows if r["arm"] == "rewire1.0"), None)
    advantage = None
    if real is not None and rewired is not None:
        diff = real["net_returns"][: min(len(real["net_returns"]), len(rewired["net_returns"]))] \
             - rewired["net_returns"][: min(len(real["net_returns"]), len(rewired["net_returns"]))]
        lo, hi = validation.block_bootstrap_ci(diff)
        advantage = {
            "sharpe_real": real["sharpe"],
            "sharpe_rewired": rewired["sharpe"],
            "mean_diff_bp": float(diff.mean() * 1e4),
            "ci95_bp": [lo * 1e4 if lo == lo else None, hi * 1e4 if hi == hi else None],
            "excludes_zero": bool(lo == lo and (lo > 0 or hi < 0)),
        }

    base = readout.baselines(
        feat0["ret_fwd"].to_numpy(), np.ones(n, dtype=bool), PERIODS_PER_YEAR
    )

    REPORTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = {
        "generated": stamp,
        "commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "config": vars(args) | {"cost": COST, "periods_per_year": PERIODS_PER_YEAR},
        "windows": n,
        "folds": len(folds),
        "baselines": base,
        "wiring_advantage": advantage,
        "deflated_sharpe_best_arm": {"arm": best["arm"], **dsr},
        "arms": [{k: v for k, v in r.items() if k != "net_returns"} for r in rows],
    }
    out = REPORTS / f"run_{stamp}.json"
    out.write_text(json.dumps(report, indent=2, default=float))

    print(f"\nbaselines: buy&hold Sharpe {base['buy_and_hold']['sharpe']:+.2f}")
    if advantage:
        print(f"wiring advantage: {advantage['mean_diff_bp']:+.4f} bp per decision, "
              f"95% CI [{advantage['ci95_bp'][0]:+.4f}, {advantage['ci95_bp'][1]:+.4f}], "
              f"excludes zero: {advantage['excludes_zero']}")
    print(f"deflated Sharpe (best arm {best['arm']}): {dsr['dsr']:.3f} "
          f"(needs > 0.95), luck benchmark {dsr['sr0_annual']:+.2f}")
    print(f"report: {out}")


if __name__ == "__main__":
    main()
