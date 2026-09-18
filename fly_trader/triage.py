"""Cheap classifier that decides which headlines deserve an LLM call.

Fetching the full history yields about 5.35M relevant articles. Scoring every
one with Jev costs only $13 but takes 74 hours at the rate limit, so throughput
is the binding constraint, not money.

The fix is the standard cascade: a cheap local model reads everything and
forwards the small fraction worth the expensive call. Jev supplies the labels
for a sample, the classifier learns to imitate its triage, and then runs over
millions of rows in seconds.

What matters is not accuracy but *recall of the interesting tail*: missing a
routine story costs nothing, missing the one that moved the market costs the
whole point. The keep-rate curve in `coverage_report` is the thing to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score

MODEL_DIR = Path("data/news")


def is_interesting(scored: pd.DataFrame, shock_at: float = 0.85, move_at: float = 1.0):
    """What the classifier is trained to find: the items Jev thought mattered.

    Either a sudden shock, or a clear directional call on the currency. Routine
    commentary that Jev rated neutral and slow is what we want to skip.
    """
    shock = scored["jev_shock"].fillna(0) >= shock_at
    directional = (scored["jev_yen"].fillna(2.0) - 2.0).abs() >= move_at
    return (shock | directional).to_numpy()


@dataclass
class Triage:
    vec_word: TfidfVectorizer
    vec_theme: TfidfVectorizer
    model: object

    def features(self, df: pd.DataFrame, fit: bool = False):
        text = df["headline"].fillna("")
        themes = (df["themes"].fillna("") + " " + df["orgs"].fillna("")).str.replace(
            ";", " ", regex=False
        )
        if fit:
            X1 = self.vec_word.fit_transform(text)
            X2 = self.vec_theme.fit_transform(themes)
        else:
            X1 = self.vec_word.transform(text)
            X2 = self.vec_theme.transform(themes)
        dense = np.column_stack(
            [
                df["gdelt_tone"].fillna(0).to_numpy(),
                df["gdelt_polarity"].fillna(0).to_numpy(),
                df["headline"].fillna("").str.split().str.len().to_numpy(),
            ]
        ).astype(np.float32)
        return sparse.hstack([X1, X2, sparse.csr_matrix(dense)]).tocsr()

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.features(df))[:, 1]

    def select(self, df: pd.DataFrame, per_window: int = 6, window: str = "15min"):
        """Keep the most promising few per decision window.

        Selecting per window rather than globally stops a single loud day from
        consuming the entire budget and leaving quiet weeks unscored.
        """
        out = df.copy()
        out["triage"] = self.predict(out)
        out["_w"] = pd.to_datetime(out["t"]).dt.floor(window)
        keep = (
            out.sort_values("triage", ascending=False)
            .groupby("_w", sort=False)
            .head(per_window)
            .index
        )
        return out.loc[sorted(keep)].drop(columns="_w")


def train(scored: pd.DataFrame, seed: int = 0) -> tuple[Triage, dict]:
    """Learn to imitate Jev's triage from a sample it has already scored."""
    y = is_interesting(scored)
    n_train = int(len(scored) * 0.7)

    t = Triage(
        vec_word=TfidfVectorizer(
            max_features=60000, ngram_range=(1, 2), min_df=2, sublinear_tf=True
        ),
        vec_theme=TfidfVectorizer(max_features=20000, min_df=2),
        model=None,
    )
    X = t.features(scored, fit=True)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]

    base = SGDClassifier(loss="log_loss", alpha=1e-5, max_iter=400,
                         tol=1e-4, random_state=seed)
    t.model = CalibratedClassifierCV(base, cv=3, method="isotonic")
    t.model.fit(Xtr, ytr)

    p = t.model.predict_proba(Xte)[:, 1]
    stats = {
        "n_train": int(n_train),
        "n_test": int(len(yte)),
        "base_rate": float(y.mean()),
        "average_precision": float(average_precision_score(yte, p)),
        "coverage": coverage_report(yte, p),
    }
    return t, stats


def coverage_report(y: np.ndarray, p: np.ndarray, rates=(0.01, 0.02, 0.05, 0.1, 0.2)):
    """How much of the interesting tail survives each keep-rate.

    Read this, not accuracy: the question is what fraction of the items that
    mattered we still send to the expensive model once we throw most away.
    """
    order = np.argsort(-p)
    total = max(1, y.sum())
    out = {}
    for r in rates:
        k = max(1, int(len(y) * r))
        out[f"keep_{int(r * 100)}pct"] = {
            "recall": float(y[order[:k]].sum() / total),
            "precision": float(y[order[:k]].mean()),
        }
    return out


def save(t: Triage, name: str = "triage") -> Path:
    import joblib

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.joblib"
    joblib.dump(t, path)
    return path


def load(name: str = "triage") -> Triage:
    import joblib

    return joblib.load(MODEL_DIR / f"{name}.joblib")
