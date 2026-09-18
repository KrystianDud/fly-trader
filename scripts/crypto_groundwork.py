"""Is BTC tradeable at our horizons? Decide before building anything.

The FX attempt failed on arithmetic that was knowable in advance: signal of
about 1 bp against costs of 0.6 bp. Nothing downstream could have rescued that.
So this runs the same arithmetic for BTC *first*, against thresholds written
down before the data is seen.

  GATE 1  edge headroom     a horizon must offer >= 4x the round-trip cost in
                            typical absolute move, or costs dominate
  GATE 2  structure         returns must show detectable departure from a
                            random walk: variance ratio outside its confidence
                            band, or significant Ljung-Box at some horizon
  GATE 3  feature edge      at least one simple feature with |IC| >= 0.02 whose
                            bootstrap CI excludes zero
  GATE 4  regime lift       predictability must concentrate somewhere useful:
                            best regime IC >= 2x the unconditional IC

Clearing 1 and 2 means the venue is viable. Clearing 3 and 4 means we have
somewhere to start. Failing 1 or 2 stops the project here, cheaply.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path("reports")
GATES = {"headroom_x": 4.0, "feature_ic": 0.02, "regime_lift_x": 2.0}


def load(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/crypto/{symbol}_1m.parquet").set_index("t").sort_index()
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def data_quality(df: pd.DataFrame) -> dict:
    full = pd.date_range(df.index.min(), df.index.max(), freq="1min")
    missing = len(full) - len(df)
    ret = np.log(df.close).diff()
    return {
        "bars": int(len(df)),
        "from": str(df.index.min()),
        "to": str(df.index.max()),
        "years": round((df.index.max() - df.index.min()).days / 365.25, 2),
        "missing_minutes": int(missing),
        "missing_pct": round(missing / len(full) * 100, 3),
        "zero_volume_bars_pct": round((df.volume == 0).mean() * 100, 3),
        "abs_return_p9999_bp": round(float(ret.abs().quantile(0.9999) * 1e4), 1),
    }


def horizons(df: pd.DataFrame, cost_bp: float) -> pd.DataFrame:
    """Absolute move per horizon against round-trip cost: the headroom test."""
    rows = []
    for mins in (1, 5, 15, 60, 240, 1440):
        px = df.close.resample(f"{mins}min").last().dropna()
        r = np.log(px).diff().dropna()
        median_abs_bp = float(r.abs().median() * 1e4)
        rows.append({
            "minutes": mins,
            "bars_per_year": int(525600 / mins),
            "sd_bp": round(float(r.std() * 1e4), 1),
            "median_abs_bp": round(median_abs_bp, 1),
            "kurtosis": round(float(stats.kurtosis(r)), 1),
            "headroom_x": round(median_abs_bp / cost_bp, 2),
            "passes_gate1": median_abs_bp / cost_bp >= GATES["headroom_x"],
        })
    return pd.DataFrame(rows)


def variance_ratio(r: np.ndarray, q: int) -> tuple[float, float]:
    """Lo-MacKinlay: VR = 1 under a random walk. Returns (VR, z) with
    heteroskedasticity-robust standard errors, because crypto is anything but
    homoskedastic."""
    n = len(r)
    mu = r.mean()
    var1 = ((r - mu) ** 2).sum() / (n - 1)
    rq = np.convolve(r, np.ones(q), "valid")
    varq = ((rq - q * mu) ** 2).sum() / (q * (n - q + 1) * (1 - 1 / (n / q)))
    vr = varq / var1

    delta = 0.0
    for j in range(1, q):
        d = (((r[j:] - mu) ** 2) * ((r[:-j] - mu) ** 2)).sum() / (var1 * (n - 1)) ** 2
        delta += (2 * (q - j) / q) ** 2 * d * (n - 1)
    z = (vr - 1) / np.sqrt(max(delta, 1e-12))
    return float(vr), float(z)


def structure(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mins in (1, 5, 15, 60):
        px = df.close.resample(f"{mins}min").last().dropna()
        r = np.log(px).diff().dropna().to_numpy()
        lb = stats.false_discovery_control if False else None  # placeholder unused
        ac1 = float(np.corrcoef(r[1:], r[:-1])[0, 1])
        vr2, z2 = variance_ratio(r, 2)
        vr5, z5 = variance_ratio(r, 5)
        # Ljung-Box on 10 lags, computed directly
        n = len(r)
        acs = [np.corrcoef(r[k:], r[:-k])[0, 1] for k in range(1, 11)]
        q_stat = n * (n + 2) * sum(a ** 2 / (n - k) for k, a in enumerate(acs, 1))
        p_lb = 1 - stats.chi2.cdf(q_stat, 10)
        rows.append({
            "minutes": mins, "autocorr_1": round(ac1, 4),
            "VR_q2": round(vr2, 3), "z_q2": round(z2, 2),
            "VR_q5": round(vr5, 3), "z_q5": round(z5, 2),
            "ljung_box_p": round(float(p_lb), 5),
            "random_walk_rejected": bool(abs(z2) > 1.96 or abs(z5) > 1.96 or p_lb < 0.05),
        })
    return pd.DataFrame(rows)


def features(df: pd.DataFrame, mins: int) -> tuple[pd.DataFrame, np.ndarray]:
    """Simple, economically motivated features. Nothing exotic on purpose:
    if none of these show anything, a clever feature is unlikely to save it."""
    g = pd.DataFrame(index=df.resample(f"{mins}min").last().index)
    px = df.close.resample(f"{mins}min").last()
    hi = df.high.resample(f"{mins}min").max()
    lo = df.low.resample(f"{mins}min").min()
    vol = df.volume.resample(f"{mins}min").sum()
    taker = (df.taker_buy_share * df.volume).resample(f"{mins}min").sum() / vol.replace(0, np.nan)
    trades = df.trades.resample(f"{mins}min").sum()

    lr = np.log(px).diff()
    sd = lr.rolling(96).std()

    g["momentum_1"] = lr / sd
    g["momentum_4"] = (np.log(px) - np.log(px.shift(4))) / (sd * 2)
    g["momentum_24"] = (np.log(px) - np.log(px.shift(24))) / (sd * np.sqrt(24))
    g["vol_ratio"] = sd / sd.rolling(96).median() - 1
    g["range_position"] = (px - lo.rolling(24).min()) / (
        hi.rolling(24).max() - lo.rolling(24).min()
    ) - 0.5
    g["order_flow"] = taker - 0.5                      # aggressive buy pressure
    g["order_flow_4"] = (taker - 0.5).rolling(4).mean()
    g["volume_surge"] = vol / vol.rolling(96).median() - 1
    g["trade_size"] = (vol / trades.replace(0, np.nan))
    g["trade_size"] = g["trade_size"] / g["trade_size"].rolling(96).median() - 1

    fwd = (np.log(px).shift(-1) - np.log(px)).to_numpy()
    return g.replace([np.inf, -np.inf], np.nan), fwd


def block_bootstrap_ic(x: np.ndarray, y: np.ndarray, block: int = 96,
                       n_boot: int = 400, seed: int = 0) -> tuple[float, float, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ic = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < block * 4:
        return ic, np.nan, np.nan
    out = np.empty(n_boot)
    nb = n // block
    for b in range(n_boot):
        starts = rng.integers(0, n - block, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])
        out[b] = stats.spearmanr(x[idx], y[idx]).statistic
    return ic, float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--cost-bp", type=float, default=10.0,
                    help="round-trip: taker fee both sides plus spread/slippage")
    ap.add_argument("--horizon", type=int, default=15)
    args = ap.parse_args()

    df = load(args.symbol)
    report = {"symbol": args.symbol, "cost_bp": args.cost_bp}

    print("═══ 1. DATA QUALITY")
    q = data_quality(df)
    report["data"] = q
    for k, v in q.items():
        print(f"   {k:24s} {v}")

    print("\n═══ 2. GATE 1 — edge headroom per horizon")
    h = horizons(df, args.cost_bp)
    report["horizons"] = h.to_dict("records")
    print(h.to_string(index=False))
    gate1 = bool(h.passes_gate1.any())

    print("\n═══ 3. GATE 2 — is it a random walk?")
    s = structure(df)
    report["structure"] = s.to_dict("records")
    print(s.to_string(index=False))
    gate2 = bool(s.random_walk_rejected.any())

    print(f"\n═══ 4. GATE 3 — feature edge at {args.horizon}m")
    g, fwd = features(df, args.horizon)
    rows = []
    for c in g.columns:
        ic, lo, hi = block_bootstrap_ic(g[c].to_numpy(), fwd)
        rows.append({"feature": c, "IC": round(ic, 4),
                     "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)})
    f = pd.DataFrame(rows).sort_values("IC", key=abs, ascending=False)
    report["features"] = f.to_dict("records")
    print(f.to_string(index=False))
    gate3 = bool(((f.IC.abs() >= GATES["feature_ic"]) & f.excludes_zero).any())

    print("\n═══ 5. GATE 4 — does predictability concentrate in a regime?")
    best = f.iloc[0].feature
    x = g[best].to_numpy()
    volq = pd.qcut(g["vol_ratio"].rank(method="first"), 4, labels=False).to_numpy()
    uncond = abs(stats.spearmanr(x[np.isfinite(x) & np.isfinite(fwd)],
                                 fwd[np.isfinite(x) & np.isfinite(fwd)]).statistic)
    reg = []
    for qi in range(4):
        m = (volq == qi) & np.isfinite(x) & np.isfinite(fwd)
        if m.sum() > 500:
            reg.append({"vol_quartile": qi + 1, "n": int(m.sum()),
                        "IC": round(float(stats.spearmanr(x[m], fwd[m]).statistic), 4)})
    r = pd.DataFrame(reg)
    report["regimes"] = {"feature": best, "unconditional_IC": round(uncond, 4),
                         "by_vol_quartile": r.to_dict("records")}
    print(f"   feature: {best}, unconditional |IC| {uncond:.4f}")
    print(r.to_string(index=False))
    gate4 = bool(len(r) and r.IC.abs().max() >= GATES["regime_lift_x"] * uncond)

    print("\n═══ VERDICT")
    for name, passed, detail in [
        ("GATE 1 headroom", gate1, f"some horizon offers >= {GATES['headroom_x']}x cost"),
        ("GATE 2 structure", gate2, "random walk rejected somewhere"),
        ("GATE 3 feature edge", gate3, f"a feature with |IC| >= {GATES['feature_ic']}, CI excluding zero"),
        ("GATE 4 regime lift", gate4, f"best regime >= {GATES['regime_lift_x']}x unconditional"),
    ]:
        print(f"   {'PASS' if passed else 'FAIL'}  {name:20s} {detail}")

    report["gates"] = {"headroom": gate1, "structure": gate2,
                       "feature_edge": gate3, "regime_lift": gate4}
    verdict = ("proceed" if gate1 and gate2 else "stop: venue not viable")
    if gate1 and gate2 and not (gate3 or gate4):
        verdict = "venue viable, but no simple edge found — needs better features, not more plumbing"
    report["verdict"] = verdict
    print(f"\n   → {verdict}")

    OUT.mkdir(exist_ok=True)
    (OUT / f"groundwork_{args.symbol}.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"   report: reports/groundwork_{args.symbol}.json")


if __name__ == "__main__":
    main()
