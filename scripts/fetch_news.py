"""Ingest the full point-in-time news history, one month per file.

Resumable: months already on disk are skipped, so this can be killed and
restarted without losing work. Monthly files also keep memory flat — the whole
history is millions of rows and does not want to live in one frame.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from fly_trader import news

OUT = Path("data/news/raw")


def months(start: datetime, end: datetime):
    m = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    while m <= end:
        nxt = (m.replace(day=28) + pd.Timedelta(days=4)).replace(day=1)
        yield m, min(nxt - pd.Timedelta(minutes=15), end)
        m = nxt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-16")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    for m_start, m_end in months(start, end):
        path = OUT / f"gdelt_{m_start:%Y-%m}.parquet"
        if path.exists():
            print(f"{m_start:%Y-%m}: already have {path.stat().st_size / 1e6:.0f} MB",
                  flush=True)
            continue
        t0 = pd.Timestamp.now()
        df = news.ingest(m_start, m_end, workers=args.workers)
        if len(df):
            df.to_parquet(path, index=False)
        took = (pd.Timestamp.now() - t0).total_seconds()
        print(f"{m_start:%Y-%m}: {len(df):>7,} articles in {took / 60:5.1f} min "
              f"-> {path.name}", flush=True)

    files = sorted(OUT.glob("*.parquet"))
    total = sum(len(pd.read_parquet(f, columns=["url"])) for f in files)
    print(f"\ndone: {total:,} articles across {len(files)} months")


if __name__ == "__main__":
    main()
