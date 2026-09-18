"""Live view of the news pipeline, stage by stage.

The pipeline has three stages and each can fail differently, so the page shows
all three rather than only the result:

  1. ingested  — everything GDELT published that passed the relevance filter
  2. triaged   — what the local classifier scored, and what it chose to forward
  3. scored    — what Jev said about the ones that got through

Run:  PYTHONPATH=. uv run uvicorn server.news_live:app --port 8138
Then: http://localhost:8138/news
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from fly_trader import news, rss, triage

POLL_SECONDS = 45
KEEP_PER_SLICE = 6          # how many survive triage per 15-minute GDELT slice
RSS_SECONDS = 20            # RSS publishes continuously, so poll it hard
RSS_MAX_PER_POLL = 8        # cap the LLM calls when a wire dumps a batch
HISTORY = 400               # rows held in memory for late joiners

app = FastAPI()
clients: set[WebSocket] = set()
history: list[dict] = []
state = {
    "last_slice": None,
    "model": None,
    "feeds": 0,
    "counts": {"ingested": 0, "triaged": 0, "scored": 0},
}


def load_model():
    try:
        return triage.load()
    except Exception:
        return None


async def broadcast(event: dict) -> None:
    history.append(event)
    del history[:-HISTORY]
    dead = []
    for ws in clients:
        try:
            await ws.send_text(json.dumps(event))
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


BACKFILL_SLICES = 4  # an hour of history on startup, so the page is not blank


async def pipeline_loop() -> None:
    state["model"] = load_model()
    await asyncio.sleep(1)
    done: set[datetime] = set()
    first = True

    while True:
        try:
            newest = await asyncio.to_thread(news.newest_slice)
            if newest is None:
                await broadcast({"type": "slice", "t": datetime.now(timezone.utc).isoformat(),
                                 "stage": "GDELT unreachable, retrying"})
            else:
                # on the first pass, fill the page with the last hour
                wanted = (
                    [newest - timedelta(minutes=15 * i) for i in range(BACKFILL_SLICES - 1, -1, -1)]
                    if first else [newest]
                )
                first = False
                for when in wanted:
                    if when in done:
                        continue
                    got = await run_slice(when)
                    if got:  # an empty slice is retried later, not marked done
                        done.add(when)
                        state["last_slice"] = when
                await broadcast({"type": "waiting", "next_after": newest.isoformat()})
        except Exception as e:  # never let one bad slice kill the loop
            await broadcast({"type": "error", "message": str(e)[:200]})
        await asyncio.sleep(POLL_SECONDS)


async def run_slice(when: datetime) -> bool:
    await broadcast({"type": "slice", "t": when.isoformat(), "stage": "fetching"})

    df = await asyncio.to_thread(news.fetch_slice, when)
    if not len(df):
        await broadcast({"type": "slice", "t": when.isoformat(), "stage": "not published yet"})
        return False

    df = df[df.headline.map(news.is_readable)]
    df = df[df.apply(news.JPY.matches, axis=1)] if len(df) else df
    if len(df):  # syndicated copies share a headline across many URLs
        df = df.assign(_k=df.headline.str.lower().str.replace(r"\s+", " ", regex=True))
        df = df.drop_duplicates(subset="_k").drop(columns="_k")
    if not len(df):
        await broadcast({"type": "slice", "t": when.isoformat(), "stage": "nothing relevant"})
        return True

    model = state["model"]
    if model is not None:
        df = df.copy()
        df["triage"] = await asyncio.to_thread(model.predict, df)
    else:
        df["triage"] = float("nan")
    df = df.sort_values("triage", ascending=False)

    state["counts"]["ingested"] += len(df)
    for _, r in df.iterrows():
        await broadcast(
            {
                "type": "article",
                "channel": "gdelt",
                "t": when.isoformat(),
                "source": r["source"],
                "headline": r["headline"],
                "tone": round(float(r["gdelt_tone"]), 2),
                "triage": None if pd.isna(r["triage"]) else round(float(r["triage"]), 3),
                "url": r["url"],
            }
        )

    chosen = df.head(KEEP_PER_SLICE)
    state["counts"]["triaged"] += len(chosen)
    for _, r in chosen.iterrows():
        await broadcast(
            {
                "type": "forwarded",
                "channel": "gdelt",
                "t": when.isoformat(),
                "source": r["source"],
                "headline": r["headline"],
                "triage": None if pd.isna(r["triage"]) else round(float(r["triage"]), 3),
            }
        )
    scored = await asyncio.to_thread(news.score, chosen, 8)

    for _, r in scored.iterrows():
        if pd.isna(r.get("jev_yen")):
            continue
        state["counts"]["scored"] += 1
        await broadcast(
            {
                "type": "scored",
                "channel": "gdelt",
                "t": when.isoformat(),
                "headline": r["headline"],
                "tone": round(float(r["gdelt_tone"]), 2),
                "triage": None if pd.isna(r.get("triage")) else round(float(r["triage"]), 3),
                "yen": round(float(r["jev_yen"]), 2),
                "conf": round(float(r["jev_yen_conf"]), 2),
                "shock": round(float(r["jev_shock"]), 2),
                "kind": r["jev_kind"],
            }
        )
    await broadcast({"type": "counts", **state["counts"]})
    return True


async def rss_loop() -> None:
    """Continuous headlines between GDELT's quarter-hourly snapshots."""
    reader = rss.Reader()
    health = await asyncio.to_thread(reader.check)
    state["feeds"] = int(health.alive.sum())
    await asyncio.to_thread(reader.prime)  # do not replay the existing backlog
    await broadcast(
        {"type": "feeds", "alive": state["feeds"], "total": int(len(health))}
    )

    while True:
        try:
            df = await asyncio.to_thread(reader.poll)
            if len(df):
                await handle_live(df)
        except Exception as e:
            await broadcast({"type": "error", "message": "rss: " + str(e)[:180]})
        await asyncio.sleep(RSS_SECONDS)


async def handle_live(df) -> None:
    """Same pipeline as a GDELT slice: score, forward the best, ask Jev."""
    model = state["model"]
    df = df.copy()
    df["triage"] = (
        await asyncio.to_thread(model.predict, df) if model is not None else float("nan")
    )
    df = df.sort_values("triage", ascending=False)

    state["counts"]["ingested"] += len(df)
    for _, r in df.iterrows():
        await broadcast(
            {
                "type": "article",
                "channel": "rss",
                "t": r["t"].isoformat(),
                "source": r["source"],
                "headline": r["headline"],
                "tone": 0.0,
                "triage": None if pd.isna(r["triage"]) else round(float(r["triage"]), 3),
                "url": r["url"],
            }
        )

    chosen = df.head(RSS_MAX_PER_POLL)
    state["counts"]["triaged"] += len(chosen)
    for _, r in chosen.iterrows():
        await broadcast(
            {
                "type": "forwarded",
                "channel": "rss",
                "t": r["t"].isoformat(),
                "source": r["source"],
                "headline": r["headline"],
                "triage": None if pd.isna(r["triage"]) else round(float(r["triage"]), 3),
            }
        )

    scored = await asyncio.to_thread(news.score, chosen, 8)
    for _, r in scored.iterrows():
        if pd.isna(r.get("jev_yen")):
            continue
        state["counts"]["scored"] += 1
        await broadcast(
            {
                "type": "scored",
                "channel": "rss",
                "t": r["t"].isoformat(),
                "headline": r["headline"],
                "tone": 0.0,
                "triage": None if pd.isna(r.get("triage")) else round(float(r["triage"]), 3),
                "yen": round(float(r["jev_yen"]), 2),
                "conf": round(float(r["jev_yen_conf"]), 2),
                "shock": round(float(r["jev_shock"]), 2),
                "kind": r["jev_kind"],
            }
        )
    await broadcast({"type": "counts", **state["counts"]})


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(pipeline_loop())
    asyncio.create_task(rss_loop())


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    clients.add(websocket)
    try:
        for event in history[-120:]:
            await websocket.send_text(json.dumps(event))
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(websocket)


@app.get("/news", response_class=HTMLResponse)
async def page() -> str:
    return (Path(__file__).parent / "news.html").read_text()


@app.get("/health")
async def health() -> dict:
    return {
        "last_slice": state["last_slice"].isoformat() if state["last_slice"] else None,
        "classifier": state["model"] is not None,
        "live_feeds": state["feeds"],
        "clients": len(clients),
        **state["counts"],
    }
