"""Pull bulk crypto history from Binance's public data archive.

No key, no rate limit, no scraping: monthly zips of 1-minute candles going back
to 2017. Crucially the kline format carries taker-buy volume alongside total
volume, so order-flow imbalance comes free without the 567 MB/month trade
archive.

    PYTHONPATH=. uv run python scripts/fetch_crypto.py --symbol BTCUSDT
"""

import argparse
import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests

BASE = "https://data.binance.vision/data/spot/monthly/klines"
OUT = Path("data/crypto")

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]


def months(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch_month(symbol: str, interval: str, y: int, m: int) -> pd.DataFrame:
    url = f"{BASE}/{symbol}/{interval}/{symbol}-{interval}-{y}-{m:02d}.zip"
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        return pd.DataFrame()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        df = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=COLUMNS)

    # some months are microseconds, some milliseconds; detect rather than assume
    unit = "us" if df.open_time.iloc[0] > 1e14 else "ms"
    df["t"] = pd.to_datetime(df.open_time, unit=unit, utc=True)

    out = df[["t", "open", "high", "low", "close", "volume", "trades"]].copy()
    # share of volume that was aggressive buying: free order-flow imbalance
    out["taker_buy_share"] = (df.taker_buy_base / df.volume.replace(0, pd.NA)).fillna(0.5)
    out["quote_volume"] = df.quote_volume
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default=date.today().strftime("%Y-%m"))
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    s = date(*map(int, args.start.split("-")), 1)
    e = date(*map(int, args.end.split("-")), 1)

    frames, missing = [], 0
    for y, m in months(s, e):
        path = OUT / f"{args.symbol}_{args.interval}_{y}-{m:02d}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
            continue
        df = fetch_month(args.symbol, args.interval, y, m)
        if not len(df):
            missing += 1
            continue
        df.to_parquet(path, index=False)
        frames.append(df)
        print(f"  {y}-{m:02d}: {len(df):>6,} bars", flush=True)

    if not frames:
        raise SystemExit("nothing fetched")
    all_bars = pd.concat(frames).drop_duplicates("t").sort_values("t")
    combined = OUT / f"{args.symbol}_{args.interval}.parquet"
    all_bars.to_parquet(combined, index=False)

    span = all_bars.t.max() - all_bars.t.min()
    gaps = len(pd.date_range(all_bars.t.min(), all_bars.t.max(), freq="1min")) - len(all_bars)
    print(f"\n{len(all_bars):,} bars, {all_bars.t.min():%Y-%m-%d} to {all_bars.t.max():%Y-%m-%d} "
          f"({span.days} days)")
    print(f"missing months: {missing}  missing minutes: {gaps:,} "
          f"({gaps / max(1, len(all_bars)) * 100:.2f}%)")
    print(f"taker buy share: mean {all_bars.taker_buy_share.mean():.3f} "
          f"(0.5 = balanced)")
    print(f"wrote {combined} ({combined.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
