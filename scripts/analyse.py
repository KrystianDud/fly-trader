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


def run_arm(tag: str, folds, weights, alpha: float, bins: str, sizes: list[int],
            extra: np.ndarray | None = None):
    acts, feat, meta = load(tag)
    X = readout.design_matrix(acts, bins=bins)
    del acts

    # most descending neurons never fire under our sensory drive, and a column
    # of zeros is 92,030 useless float32s. Dropping them takes the design
    # matrix from ~1.9 GB to ~0.6 GB and changes no result.
    alive = X.std(axis=0) > 0
    X = np.ascontiguousarray(X[:, alive])
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
    metrics["live_features"] = int(alive.sum())

    # does the brain add anything on top of the features it was handed?
    if extra is not None:
        Xboth = np.hstack([X, extra.astype(np.float32)])
        rb = readout.fit(Xboth, y, folds, alpha=alpha, weights=weights)
        mb = readout.evaluate(rb, y, cost=COST, periods_per_year=PERIODS_PER_YEAR)
        metrics["with_features"] = {k: v for k, v in mb.items() if k != "net_returns"}
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

    # The comparisons that make the brain arms interpretable. Without these,
    # a zero result is ambiguous: it could mean the market is unpredictable, or
    # that the brain discarded signal it was handed.
    FEATURE_COLS = ["fast_up", "fast_down", "slow_up", "slow_down",
                    "volatility", "looming"]
    rows = []

    Xf = feat0[FEATURE_COLS].to_numpy(dtype=np.float32)
    yf = feat0["ret_fwd"].to_numpy(dtype=np.float64)
    ref = readout.fit(Xf, yf, folds, alpha=1.0, weights=weights)
    m = readout.evaluate(ref, yf, cost=COST, periods_per_year=PERIODS_PER_YEAR)
    m |= {"arm": "features_only", "reciprocity": float("nan"),
          "windows": n, "live_features": Xf.shape[1], "learning_curve": []}
    rows.append(m)
    print(f"{'features_only':12s} IC {m['ic']:+.4f}  hit {m['hit_rate']:.3f}  "
          f"net {m['net_mean_bp']:+.3f} bp  Sharpe {m['sharpe']:+.2f}  "
          f"trades {m['trade_share']:.2f}", flush=True)

    for tag in tags:
        m = run_arm(tag, folds, weights, args.alpha, args.bins, sizes,
                    extra=Xf if tag.endswith(f"real_s{args.seed}_{args.sim_ms}ms") else None)
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

    lines = [
        f"# Run {stamp}",
        "",
        f"{n:,} decisions, {len(folds)} purged walk-forward folds, "
        f"costs {COST * 1e4:.1f} bp per position change.",
        "",
        "## Dose-response: performance against how much structure was destroyed",
        "",
        "| Arm | Reciprocity | IC | Hit rate | Net bp | Sharpe | Trades |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: -r["reciprocity"]):
        lines.append(
            f"| {r['arm']} | {r['reciprocity']:.4f} | {r['ic']:+.4f} | "
            f"{r['hit_rate']:.3f} | {r['net_mean_bp']:+.3f} | {r['sharpe']:+.2f} | "
            f"{r['trade_share']:.2f} |"
        )
    lines += ["", "## Learning curves (IC by training size)", "",
              "| Arm | " + " | ".join(str(s) for s in sizes) + " |",
              "|---" * (len(sizes) + 1) + "|"]
    for r in sorted(rows, key=lambda r: -r["reciprocity"]):
        by = {c["train_size"]: c["ic"] for c in r["learning_curve"]}
        lines.append(f"| {r['arm']} | " +
                     " | ".join(f"{by.get(s, float('nan')):+.4f}" for s in sizes) + " |")
    lines += ["", "## Does the brain add anything to its own inputs?", ""]
    r_real = next((r for r in rows if r["arm"] == "real"), None)
    r_feat = next((r for r in rows if r["arm"] == "features_only"), None)
    if r_real and r_feat:
        lines += [
            "| Readout input | IC | Hit rate | Sharpe |",
            "|---|---|---|---|",
            f"| raw features only | {r_feat['ic']:+.4f} | {r_feat['hit_rate']:.3f} | "
            f"{r_feat['sharpe']:+.2f} |",
            f"| descending neurons only | {r_real['ic']:+.4f} | {r_real['hit_rate']:.3f} | "
            f"{r_real['sharpe']:+.2f} |",
        ]
        wf = r_real.get("with_features")
        if wf:
            lines.append(
                f"| both | {wf['ic']:+.4f} | {wf['hit_rate']:.3f} | {wf['sharpe']:+.2f} |"
            )
    lines += ["", "## Verdict", ""]
    if advantage:
        lines += [
            f"- Wiring advantage: **{advantage['mean_diff_bp']:+.4f} bp** per decision, "
            f"95% CI [{advantage['ci95_bp'][0]:+.4f}, {advantage['ci95_bp'][1]:+.4f}]",
            f"- Excludes zero: **{advantage['excludes_zero']}**",
        ]
    lines += [
        f"- Best arm `{best['arm']}` deflated Sharpe **{dsr['dsr']:.3f}** "
        f"(needs > 0.95), luck benchmark {dsr['sr0_annual']:+.2f}",
        f"- Buy and hold Sharpe {base['buy_and_hold']['sharpe']:+.2f}",
    ]
    (REPORTS / f"run_{stamp}.md").write_text("\n".join(lines) + "\n")

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
