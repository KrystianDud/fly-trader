"""Continuous live headlines from RSS, to complement GDELT's 15-minute cadence.

GDELT publishes one snapshot per quarter hour and nothing arrives between them.
RSS publishes as stories appear, so this is what gives the live view a pulse.

The two sources do different jobs and neither replaces the other:

  GDELT  archived, point-in-time, the only option for backfilling history,
         but headlines are recovered from URL slugs and so are lossy
  RSS     seconds-fresh and carries the real headline and summary,
         but has no past: a feed only ever shows you now

Feeds are deliberately a mix of wires, market desks and Japan-specific outlets,
since a yen story often breaks in Tokyo before it reaches a global wire.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

def google_news(query: str) -> str:
    """Google News as a query-able firehose.

    One of these aggregates thousands of publishers and updates continuously,
    which is the cheapest way to widen coverage without babysitting a hundred
    individual feeds.
    """
    from urllib.parse import quote_plus

    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=en-US&gl=US&ceid=US:en"
    )


FEEDS: dict[str, str] = {
    # aggregated queries: the widest net, updated continuously
    "gnews_yen": google_news("yen OR USDJPY when:1d"),
    "gnews_boj": google_news("Bank of Japan OR BOJ policy when:1d"),
    "gnews_fed": google_news("Federal Reserve OR FOMC rates when:1d"),
    "gnews_inflation": google_news("inflation CPI report when:1d"),
    "gnews_intervention": google_news("currency intervention finance ministry when:2d"),
    "gnews_carry": google_news("carry trade OR risk-off markets when:2d"),
    # market desks and wires
    "investing_fx": "https://www.investing.com/rss/news_285.rss",
    "investing_econ": "https://www.investing.com/rss/news_95.rss",
    "fxstreet": "https://www.fxstreet.com/rss/news",
    "forexlive": "https://www.forexlive.com/feed/news",
    "cnbc_econ": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    "cnbc_fx": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=19832390",
    "cnbc_top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "marketwatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "marketwatch_rt": "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
    "seekingalpha": "https://seekingalpha.com/market_currents.xml",
    "benzinga": "https://www.benzinga.com/feed",
    "bbc_business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "guardian_business": "https://www.theguardian.com/uk/business/rss",
    "ft_home": "https://www.ft.com/rss/home",
    # Japan, where a yen story often breaks first
    "japantimes": "https://www.japantimes.co.jp/feed/",
    "nikkei_asia": "https://asia.nikkei.com/rss/feed/nar",
    "nhk_business": "https://www3.nhk.or.jp/rss/news/cat5.xml",
    # central banks and statistics agencies, straight from the source
    "fed_press": "https://www.federalreserve.gov/feeds/press_all.xml",
    "fed_speeches": "https://www.federalreserve.gov/feeds/speeches.xml",
    "boj": "https://www.boj.or.jp/en/rss/whatsnew.xml",
    "bls": "https://www.bls.gov/feed/bls_latest.rss",
}


KEYWORDS = (
    "yen", "japan", "boj", "jpy", "tokyo", "ueda",
    "fed", "fomc", "federal reserve", "powell", "treasury", "yield",
    "inflation", "cpi", "payroll", "jobs", "rate", "rates", "hike", "cut",
    "dollar", "currency", "intervention", "carry",
)


@dataclass
class Reader:
    """Polls feeds and yields only what it has not seen before."""

    feeds: dict[str, str] = field(default_factory=lambda: dict(FEEDS))
    seen: set[str] = field(default_factory=set)
    max_seen: int = 20000

    def _key(self, title: str, link: str) -> str:
        norm = " ".join(title.lower().split())
        return hashlib.sha1((norm or link).encode()).hexdigest()

    def poll(self, relevant_only: bool = True, timeout: int = 10) -> pd.DataFrame:
        """One pass over every feed. Safe to call every few seconds."""
        import feedparser

        rows = []
        for name, url in self.feeds.items():
            try:
                parsed = feedparser.parse(url, request_headers={"User-Agent": "fly-trader/0.1"})
            except Exception:
                continue
            for e in parsed.entries[:40]:
                title = (getattr(e, "title", "") or "").strip()
                link = getattr(e, "link", "") or ""
                if not title:
                    continue
                key = self._key(title, link)
                if key in self.seen:
                    continue
                if relevant_only and not any(k in title.lower() for k in KEYWORDS):
                    continue

                published = getattr(e, "published_parsed", None) or getattr(
                    e, "updated_parsed", None
                )
                t = (
                    datetime(*published[:6], tzinfo=timezone.utc)
                    if published
                    else datetime.now(timezone.utc)
                )
                # ignore stale items a feed happens to still be carrying
                if datetime.now(timezone.utc) - t > timedelta(hours=6):
                    self.seen.add(key)
                    continue

                self.seen.add(key)
                rows.append(
                    {
                        "t": t,
                        "source": name,
                        "url": link,
                        "headline": title,
                        "summary": (getattr(e, "summary", "") or "")[:300],
                        "themes": "",
                        "orgs": "",
                        "gdelt_tone": 0.0,
                        "gdelt_polarity": 0.0,
                        "channel": "rss",
                    }
                )

        if len(self.seen) > self.max_seen:  # keep the dedupe set bounded
            self.seen = set(list(self.seen)[-self.max_seen // 2 :])

        return pd.DataFrame(rows).sort_values("t") if rows else pd.DataFrame()

    def check(self, timeout: int = 10) -> pd.DataFrame:
        """Which feeds are actually alive. Dead ones are dropped, loudly."""
        import feedparser

        rows = []
        for name, url in list(self.feeds.items()):
            try:
                p = feedparser.parse(url, request_headers={"User-Agent": "fly-trader/0.1"})
                n = len(p.entries)
            except Exception:
                n = 0
            rows.append({"feed": name, "entries": n, "alive": n > 0})
            if n == 0:
                self.feeds.pop(name, None)
        return pd.DataFrame(rows).sort_values("entries", ascending=False)

    def prime(self) -> int:
        """Mark everything currently in the feeds as seen, without emitting it.

        Without this, starting the reader dumps hundreds of hours-old stories
        into the live view as though they had just broken.
        """
        df = self.poll(relevant_only=False)
        return len(df)
