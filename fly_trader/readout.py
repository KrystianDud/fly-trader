"""The only trained part of the system.

A ridge regression from descending-neuron spike counts to the forward return,
with the trading threshold chosen on the training fold alone. Kept deliberately
small and heavily regularised: if this layer is powerful, the experiment stops
measuring the wiring and starts measuring the readout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.linear_model import Ridge

from .validation import Fold


@dataclass
class Result:
    pred: np.ndarray       # out-of-sample predictions, NaN where never tested
    position: np.ndarray   # -1, 0, +1
    tested: np.ndarray     # boolean mask of samples with an out-of-sample value


def design_matrix(acts: np.ndarray, bins: str = "all") -> np.ndarray:
    """(windows, bins, neurons) -> (windows, features).

    bins='all' keeps the time course; 'sum' collapses it, which is the ablation
    that tells us whether spike timing carried anything.
    """
    a = acts.astype(np.float32)
    if bins == "sum":
        a = a.sum(axis=1, keepdims=True)
    return a.reshape(len(a), -1)


def fit(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[Fold],
    alpha: float = 1e4,
    weights: np.ndarray | None = None,
    gate_quantile: float = 0.6,
) -> Result:
    """Walk-forward fit. Everything — scaling, threshold — comes from training
    data only; the test block is never consulted.
    """
    pred = np.full(len(y), np.nan)
    position = np.zeros(len(y))
    tested = np.zeros(len(y), dtype=bool)

    for f in folds:
        Xtr, ytr = X[f.train], y[f.train]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd == 0] = 1.0

        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit((Xtr - mu) / sd, ytr,
                  sample_weight=None if weights is None else weights[f.train])

        # confidence gate: trade only when the signal is larger than most of
        # what the model produced in training
        train_pred = model.predict((Xtr - mu) / sd)
        gate = np.quantile(np.abs(train_pred), gate_quantile)

        p = model.predict((X[f.test] - mu) / sd)
        pred[f.test] = p
        position[f.test] = np.where(np.abs(p) < gate, 0.0, np.sign(p))
        tested[f.test] = True

    return Result(pred=pred, position=position, tested=tested)


def evaluate(
    res: Result,
    forward_return: np.ndarray,
    cost: float,
    periods_per_year: float,
) -> dict:
    """Out-of-sample metrics. `cost` is charged whenever the position changes."""
    m = res.tested
    pos, r = res.position[m], forward_return[m]
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))
    net = pos * r - turnover * cost

    traded = pos != 0
    ic = (
        float(stats.spearmanr(res.pred[m], r).statistic)
        if m.sum() > 8 else float("nan")
    )
    hit = float(((pos * r) > 0)[traded].mean()) if traded.any() else float("nan")
    sd = net.std(ddof=1)

    return {
        "n_tested": int(m.sum()),
        "ic": ic,
        "hit_rate": hit,
        "trade_share": float(traded.mean()),
        "gross_mean_bp": float((pos * r).mean() * 1e4),
        "net_mean_bp": float(net.mean() * 1e4),
        "sharpe": float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
        "sharpe_gross": float(
            (pos * r).mean() / (pos * r).std(ddof=1) * np.sqrt(periods_per_year)
        ) if (pos * r).std(ddof=1) > 0 else 0.0,
        "max_drawdown_bp": float(_max_dd(net) * 1e4),
        "net_returns": net,
    }


def _max_dd(returns: np.ndarray) -> float:
    equity = np.cumsum(returns)
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def baselines(forward_return: np.ndarray, mask: np.ndarray, periods_per_year: float) -> dict:
    """What the fly has to beat: always long, always flat, and a coin flip."""
    r = forward_return[mask]
    out = {}
    for name, pos in {
        "buy_and_hold": np.ones(len(r)),
        "always_flat": np.zeros(len(r)),
    }.items():
        net = pos * r
        sd = net.std(ddof=1)
        out[name] = {
            "sharpe": float(net.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else 0.0,
            "net_mean_bp": float(net.mean() * 1e4),
        }
    return out
