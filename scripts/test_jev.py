"""Check the Jev connection: latency, and whether threat scores look sane."""

import time

from dotenv import load_dotenv
from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

load_dotenv()

HEADLINES = [
    "Bank of Japan unexpectedly raises rates, yen surges most since 2024",
    "US CPI comes in hotter than expected; Treasury yields spike",
    "Quiet session in Asia as traders await Friday's payrolls",
    "Japanese finance minister declines to comment on currency levels",
    "Emergency BOJ meeting called after yen slides past 165 to the dollar",
]

QUESTIONS = {
    "yen_strength": Score(
        instructions="What does this news imply for the Japanese yen against the US dollar?",
        criteria=[
            "Yen falls sharply against the dollar",
            "Yen drifts weaker",
            "No clear direction for the yen",
            "Yen drifts stronger",
            "Yen rises sharply against the dollar",
        ],
    ),
    "shock": Noul(
        instructions="Is this a sudden shock that would move markets within minutes, "
        "rather than slow-moving background news?"
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

client = TypeSafeClient()
latencies = []

for text in HEADLINES:
    t0 = time.time()
    r = client.system_one(text, QUESTIONS)
    ms = (time.time() - t0) * 1000
    latencies.append(ms)

    yen = r.scores["yen_strength"]
    print(f"\n{text[:70]}")
    print(
        f"  yen {yen.score:.2f}/4 (conf {yen.confidence:.2f})   "
        f"shock {r.nouls['shock'].noul:.2f}   "
        f"kind {r.choices['kind'].choice}   [{ms:.0f} ms]"
    )

print(f"\nmedian latency: {sorted(latencies)[len(latencies) // 2]:.0f} ms")
