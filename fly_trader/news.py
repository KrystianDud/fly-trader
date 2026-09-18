"""Point-in-time news: GDELT ingestion and semantic scoring.

GDELT publishes a global news snapshot every 15 minutes and archives it, so a
slice timestamped 12:00 contains only what existed by 12:00. That property is
the whole point: a backtest fed with headlines the model could not have seen is
worthless, and this is the cheapest honest source of timestamped text.

Headline text comes from the article URL slug, which for most news sites is the
headline with hyphens. It is imperfect and occasionally truncated, which is
worth remembering when reading any result built on it.

GDELT also ships its own lexicon tone score per article. We keep it, because it
is the free baseline that a language model has to beat to justify its place.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

GKG_URL = "https://data.gdeltproject.org/gdeltv2/{stamp}.gkg.csv.zip"
LAST_UPDATE = "https://data.gdeltproject.org/gdeltv2/lastupdate.txt"
CACHE = Path("data/news")

# GKG v2.1 columns we use (0-indexed)
COL_DATE, COL_SOURCE, COL_URL = 1, 3, 4
COL_THEMES, COL_ORGS, COL_TONE = 7, 13, 15

SLUG_NOISE = re.compile(
    r"\.(html?|php|aspx?)$|^\d+$|^(news|article|story|amp|index|videos?|live)$",
    re.I,
)


@dataclass
class Filter:
    """What counts as relevant. Keep it explicit rather than clever."""

    any_of: tuple[str, ...]
    domains: tuple[str, ...] = ()

    def matches(self, row: pd.Series) -> bool:
        """Relevance must be visible in the headline itself.

        Matching on GDELT themes instead lets in film reviews and airline
        awards, because an article tagged ECON_INFLATION somewhere in its body
        is not an article about inflation.
        """
        if self.domains and row["source"] not in self.domains:
            return False
        return any(k in row["headline"].lower() for k in self.any_of)


JPY = Filter(
    any_of=(
        "yen", "japan", "boj", "bank of japan", "jpy", "kuroda", "ueda",
        "federal reserve", "fed ", "fomc", "treasury yield", "carry trade",
        "interest rate", "inflation", "payroll",
    )
)


def slug_to_headline(url: str) -> str:
    """Recover readable text from a news URL. Imperfect by construction."""
    tail = url.rstrip("/").split("/")[-1]
    tail = re.sub(r"\.(html?|php|aspx?|amp)$", "", tail, flags=re.I)
    parts = [p for p in re.split(r"[-_]", tail) if p and not SLUG_NOISE.match(p)]
    parts = [p for p in parts if not (p.isdigit() and len(p) > 4)]
    return " ".join(parts).strip()


def is_readable(headline: str) -> bool:
    """Reject slugs that are identifiers rather than words.

    Plenty of sites use hashes or article ids in the URL, which produce
    confident-looking nonsense like "d48617fa 8a31 b048 e5e59f28cec1".
    """
    words = headline.split()
    if len(words) < 4:
        return False
    junk = sum(
        1
        for w in words
        if (any(c.isdigit() for c in w) and any(c.isalpha() for c in w))
        or len(w) > 18
    )
    return junk / len(words) < 0.3


def fetch_slice(when: datetime, timeout: int = 60) -> pd.DataFrame:
    """One 15-minute GDELT snapshot. Returns an empty frame if unavailable."""
    stamp = when.strftime("%Y%m%d%H%M%S")
    try:
        r = requests.get(GKG_URL.format(stamp=stamp), timeout=timeout)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            raw = z.read(z.namelist()[0]).decode("utf-8", errors="replace")
    except Exception:
        return pd.DataFrame()

    rows = []
    for line in raw.splitlines():
        f = line.split("\t")
        if len(f) <= COL_TONE:
            continue
        tone = f[COL_TONE].split(",")
        rows.append(
            {
                "t": when,
                "source": f[COL_SOURCE],
                "url": f[COL_URL],
                "headline": slug_to_headline(f[COL_URL]),
                "themes": f[COL_THEMES][:400],
                "orgs": f[COL_ORGS][:200],
                "gdelt_tone": float(tone[0]) if tone and tone[0] else 0.0,
                "gdelt_polarity": float(tone[3]) if len(tone) > 3 and tone[3] else 0.0,
            }
        )
    return pd.DataFrame(rows)


def newest_slice(timeout: int = 20) -> datetime | None:
    """Ask GDELT what it has just published, rather than guessing the lag.

    lastupdate.txt names the current files, so the timestamp comes from the
    source instead of from an assumption about how far behind it runs.
    """
    try:
        r = requests.get(LAST_UPDATE, timeout=timeout)
        r.raise_for_status()
        for line in r.text.splitlines():
            if ".gkg.csv.zip" in line:
                stamp = line.rsplit("/", 1)[-1].split(".")[0]
                return datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(
                    tzinfo=timezone.utc
                )
    except Exception:
        return None
    return None


def slices_between(start: datetime, end: datetime) -> list[datetime]:
    when = start.replace(minute=start.minute // 15 * 15, second=0, microsecond=0)
    out = []
    while when <= end:
        out.append(when)
        when += timedelta(minutes=15)
    return out


def ingest(
    start: datetime,
    end: datetime,
    filt: Filter = JPY,
    workers: int = 12,
    progress: bool = False,
) -> pd.DataFrame:
    """Every 15-minute slice in a range, filtered and de-duplicated.

    Fetched in parallel: sequentially, a year of slices takes about ten hours,
    almost all of it waiting on the network.
    """
    from concurrent.futures import ThreadPoolExecutor

    stamps = slices_between(start, end)
    out, missing = [], 0

    def one(when: datetime) -> pd.DataFrame:
        df = fetch_slice(when)
        if not len(df):
            return df
        df = df[df.headline.map(is_readable)]
        return df[df.apply(filt.matches, axis=1)] if len(df) else df

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, df in enumerate(pool.map(one, stamps)):
            if len(df):
                out.append(df)
            else:
                missing += 1
            if progress and i % 200 == 0:
                print(f"  {i}/{len(stamps)} slices, {missing} empty", flush=True)

    if missing:
        print(f"note: {missing} of {len(stamps)} slices returned nothing "
              f"(GDELT gaps or all-filtered)")
    if not out:
        return pd.DataFrame()
    rows = pd.concat(out, ignore_index=True)
    # syndicated copies share a headline across many URLs, so de-duplicate on
    # the text as well as the link
    rows["_key"] = rows["headline"].str.lower().str.replace(r"\s+", " ", regex=True)
    rows = rows.drop_duplicates(subset="url").drop_duplicates(subset=["_key"])
    return rows.drop(columns="_key").sort_values("t").reset_index(drop=True)


# ---------------------------------------------------------------- scoring

QUESTIONS = None  # built lazily so importing this module needs no API key


def _questions():
    from typesafe_sdk import Choice, Noul, Score

    return {
        "yen": Score(
            instructions="What does this news imply for the Japanese yen "
            "against the US dollar?",
            criteria=[
                "Yen falls sharply against the dollar",
                "Yen drifts weaker",
                "No clear direction for the yen",
                "Yen drifts stronger",
                "Yen rises sharply against the dollar",
            ],
        ),
        "shock": Noul(
            instructions="Is this a sudden shock that would move markets within "
            "minutes, rather than slow-moving background news?"
        ),
        "kind": Choice(
            instructions="What kind of news is this?",
            criteria={
                "central_bank": "Rate decisions, official statements, interventions",
                "data": "Economic releases such as inflation, jobs, growth",
                "politics": "Elections, fiscal policy, geopolitical events",
                "noise": "Routine commentary with no new information",
            },
        ),
    }


def score(headlines: pd.DataFrame, concurrency: int = 8) -> pd.DataFrame:
    """Score headlines with Jev. Ask what the text says, never what it implies:
    the model reads literally and its documented weakness is indirection.
    """
    import asyncio

    from dotenv import load_dotenv
    from typesafe_sdk import AsyncTypeSafeClient

    load_dotenv()
    qs = _questions()

    async def run() -> list[dict]:
        sem = asyncio.Semaphore(concurrency)
        async with AsyncTypeSafeClient() as client:

            async def one(text: str) -> dict:
                async with sem:
                    try:
                        r = await client.system_one(text, qs)
                        return {
                            "jev_yen": r.scores["yen"].score,
                            "jev_yen_conf": r.scores["yen"].confidence,
                            "jev_shock": r.nouls["shock"].noul,
                            "jev_kind": r.choices["kind"].choice,
                        }
                    except Exception as e:  # keep the row, mark it unscored
                        return {"jev_error": str(e)[:80]}

            return await asyncio.gather(*(one(h) for h in headlines.headline))

    scored = pd.DataFrame(asyncio.run(run()))
    return pd.concat([headlines.reset_index(drop=True), scored], axis=1)


def to_signal(scored: pd.DataFrame, grid: pd.DatetimeIndex, halflife_min: float = 45.0):
    """Collapse scored headlines onto the decision grid.

    News decays: a headline from three hours ago matters less than one from ten
    minutes ago, so each item contributes with an exponential decay. Only items
    strictly before a decision time contribute to it.
    """
    import numpy as np

    s = scored.dropna(subset=["jev_yen"]) if "jev_yen" in scored else scored
    if not len(s):
        return pd.DataFrame(index=grid, columns=["news_good", "news_bad"]).fillna(0.0)

    t = s["t"].to_numpy()
    # centre the 0-4 score on zero, weight by how shock-like the item is
    direction = (s["jev_yen"].to_numpy() - 2.0) / 2.0
    weight = 0.3 + 0.7 * s["jev_shock"].to_numpy()

    good = np.zeros(len(grid))
    bad = np.zeros(len(grid))
    lam = np.log(2) / (halflife_min * 60)
    gt = grid.to_numpy()

    for i, now in enumerate(gt):
        age = (now - t) / np.timedelta64(1, "s")
        live = (age > 0) & (age < halflife_min * 60 * 6)
        if not live.any():
            continue
        decay = np.exp(-lam * age[live]) * weight[live]
        d = direction[live] * decay
        good[i] = d[d > 0].sum()
        bad[i] = -d[d < 0].sum()

    out = pd.DataFrame({"news_good": good, "news_bad": bad}, index=grid)
    for c in out:  # squash into the 0-1 range the sensory encoder expects
        hi = out[c].quantile(0.99)
        out[c] = (out[c] / hi).clip(0, 1) if hi > 0 else 0.0
    return out
