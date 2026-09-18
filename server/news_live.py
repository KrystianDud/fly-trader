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

from fly_trader import news, triage

POLL_SECONDS = 60
KEEP_PER_SLICE = 6          # how many survive triage per 15-minute slice
HISTORY = 400               # rows held in memory for late joiners

app = FastAPI()
clients: set[WebSocket] = set()
history: list[dict] = []
state = {"last_slice": None, "model": None, "counts": {"ingested": 0, "triaged": 0, "scored": 0}}


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


def latest_slice_time() -> datetime:
    """GDELT publishes every 15 minutes with a few minutes of lag."""
    now = datetime.now(timezone.utc) - timedelta(minutes=5)
    return now.replace(minute=now.minute // 15 * 15, second=0, microsecond=0)


async def pipeline_loop() -> None:
    state["model"] = load_model()
    await asyncio.sleep(1)

    while True:
        try:
            slice_t = latest_slice_time()
            if slice_t != state["last_slice"]:
                await run_slice(slice_t)
                state["last_slice"] = slice_t
        except Exception as e:  # never let one bad slice kill the loop
            await broadcast({"type": "error", "message": str(e)[:200]})
        await asyncio.sleep(POLL_SECONDS)


async def run_slice(when: datetime) -> None:
    await broadcast({"type": "slice", "t": when.isoformat(), "stage": "fetching"})

    df = await asyncio.to_thread(news.fetch_slice, when)
    if not len(df):
        await broadcast({"type": "slice", "t": when.isoformat(), "stage": "empty"})
        return

    df = df[df.headline.map(news.is_readable)]
    df = df[df.apply(news.JPY.matches, axis=1)] if len(df) else df
    if not len(df):
        await broadcast({"type": "slice", "t": when.isoformat(), "stage": "nothing relevant"})
        return

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
    scored = await asyncio.to_thread(news.score, chosen, 8)

    for _, r in scored.iterrows():
        if pd.isna(r.get("jev_yen")):
            continue
        state["counts"]["scored"] += 1
        await broadcast(
            {
                "type": "scored",
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


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(pipeline_loop())


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
        "clients": len(clients),
        **state["counts"],
    }
