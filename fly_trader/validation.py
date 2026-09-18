"""Validation that does not lie to us.

Three ideas from Lopez de Prado's Advances in Financial Machine Learning, all
of which exist because the naive versions inflate results:

1. Purging and embargo. Our labels look forward, so a training sample whose
   label horizon overlaps the test set has seen the test period. Purge those,
   and embargo a gap afterwards to kill serial-correlation leakage.
2. Sample uniqueness. Overlapping windows are not independent observations;
   counting them as such overstates how much data we have.
3. Deflated Sharpe. Run enough configurations and one looks brilliant by
   chance. Deflating by the number of trials is what stops us believing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Fold:
    train: np.ndarray
    test: np.ndarray


def purged_walk_forward(
    n: int,
    n_folds: int = 6,
    horizon: int = 1,
    embargo: float = 0.01,
    expanding: bool = True,
) -> list[Fold]:
    """Forward-chaining folds with purging and an embargo gap.

    horizon: how many samples ahead a label looks, in samples.
    embargo: extra gap after the test block, as a fraction of the sample count.
    """
    idx = np.arange(n)
    fold_size = n // (n_folds + 1)
    gap = int(round(embargo * n))
    folds = []

    for k in range(n_folds):
        test_start = fold_size * (k + 1)
        test_end = min(n, test_start + fold_size)
        test = idx[test_start:test_end]

        train_end = test_start - horizon  # purge: labels must not reach the test block
        train_start = 0 if expanding else max(0, train_end - fold_size * 2)
        train = idx[train_start:max(train_start, train_end)]

        # embargo the samples immediately after the test block
        after = idx[test_end + gap:]
        if len(after) and not expanding:
            train = np.concatenate([train, after])

        if len(train) and len(test):
            folds.append(Fold(train=train, test=test))
    return folds


def sample_weights(n: int, horizon: int) -> np.ndarray:
    """Down-weight overlapping samples by how many labels each period shares.

    With a fixed horizon every interior sample overlaps `horizon` others, so
    this is uniform 1/horizon except at the edges; it matters more once we move
    to triple-barrier labels with variable horizons.
    """
    counts = np.zeros(n + horizon)
    for i in range(n):
        counts[i:i + horizon] += 1
    w = np.array([1.0 / np.mean(counts[i:i + horizon]) for i in range(n)])
    return w / w.mean()


def sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=float)
    if r.std(ddof=1) == 0 or len(r) < 2:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year))


def expected_max_sharpe(trial_sharpes: np.ndarray, periods_per_year: float) -> float:
    """Per-period Sharpe the best of N random strategies would reach by luck.

    Uses the spread of the Sharpes actually observed across the sweep, which is
    the honest estimate of how much variation chance alone produces here.
    """
    s = np.asarray(trial_sharpes, dtype=float) / np.sqrt(periods_per_year)
    m = max(2, len(s))
    sd = s.std(ddof=1) if len(s) > 1 else 1.0
    e = np.euler_gamma
    z1 = stats.norm.ppf(1 - 1.0 / m)
    z2 = stats.norm.ppf(1 - 1.0 / (m * np.e))
    return float(sd * ((1 - e) * z1 + e * z2))


def deflated_sharpe(
    observed_sharpe: float,
    returns: np.ndarray,
    trial_sharpes: np.ndarray,
    periods_per_year: float,
) -> dict:
    """Probability the observed Sharpe beats the best-of-N-by-luck benchmark,
    accounting for skew, fat tails and the number of configurations tried.

    Below ~0.95 means the result is not distinguishable from the luckiest of
    many random tries, however good the raw Sharpe looks.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 8 or r.std(ddof=1) == 0:
        return {"dsr": 0.0, "sr0_annual": float("nan"), "n_trials": len(trial_sharpes)}

    sr = observed_sharpe / np.sqrt(periods_per_year)  # per-period
    g1 = float(stats.skew(r))
    g2 = float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(trial_sharpes, periods_per_year)

    denom = np.sqrt(max(1e-12, 1 - g1 * sr + (g2 - 1) / 4 * sr**2))
    dsr = stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom)

    return {
        "dsr": float(dsr),
        "sr0_annual": float(sr0 * np.sqrt(periods_per_year)),
        "skew": g1,
        "kurtosis": g2,
        "n_trials": int(len(trial_sharpes)),
    }


def block_bootstrap_ci(
    values: np.ndarray, block: int = 24, n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% CI for a mean under serial correlation, by moving-block bootstrap."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < block * 2:
        return (float("nan"), float("nan"))
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    starts_max = n - block
    for b in range(n_boot):
        starts = rng.integers(0, starts_max + 1, n_blocks)
        sample = np.concatenate([v[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def assert_no_leakage(folds: list[Fold], horizon: int) -> None:
    """Run guard: any overlap between a training label window and its test block
    is a bug, and the run should stop rather than produce a flattering number.
    """
    for i, f in enumerate(folds):
        if len(f.train) == 0 or len(f.test) == 0:
            raise AssertionError(f"fold {i}: empty split")
        train_label_end = f.train.max() + horizon
        if train_label_end >= f.test.min():
            raise AssertionError(
                f"fold {i}: training labels reach index {train_label_end}, "
                f"test starts at {f.test.min()} (horizon={horizon})"
            )
        if np.intersect1d(f.train, f.test).size:
            raise AssertionError(f"fold {i}: train and test overlap")
