# Metrics, gates and service levels

Last reviewed: 2026-09-17

Three kinds of measurement, kept separate on purpose:

- **KPIs** — did the experiment answer its question, and what did it find
- **SLIs/SLOs** — is the machinery working and fast enough
- **Gates** — the conditions that must hold before the project is allowed to
  advance to the next phase, especially any phase involving money

## 1. Key performance indicators

### Primary (scientific)

| KPI | Definition | Target |
|---|---|---|
| **Wiring advantage** | Out-of-sample metric of real connectome minus degree-preserving rewire, with 95% CI from walk-forward folds | CI excluding zero, in either direction |
| **Attribution purity** | Share of trainable parameters outside the readout | 0% |
| **Replication** | Sign of the wiring advantage on a second, independent market | Same sign |

The wiring advantage is the headline number. It is reported even when negative.

### Secondary (decision quality)

| KPI | Definition | Reference |
|---|---|---|
| Information coefficient | Rank correlation between predicted and realised forward return | > 0.02 is notable at this frequency |
| Hit rate above threshold | Directional accuracy on trades taken above the confidence gate | vs 50% and vs the rewire |
| Net Sharpe | Annualised, after costs and slippage | vs buy-and-hold and vs always-flat |
| Max drawdown | Peak-to-trough on the equity curve | Reported, not optimised |
| Trade rate | Decisions acted on per day | Must stay plausible for 15-minute bars |
| Cost sensitivity | Net Sharpe recomputed at 2× assumed spread | Sign must survive |

### Tertiary (the parts that make it worth doing)

| KPI | Definition |
|---|---|
| News channel lift | Metric with the Jev channel minus without, same wiring |
| Timeframe merge benefit | Metric with both 1m and 15m channels minus each alone |
| Giant Fiber specificity | Whether DNp01 activity concentrates around genuine adverse spikes |
| Demo legibility | A non-specialist can explain what the fly is reacting to |

## 2. Service level indicators and objectives

Phase 0/1 is research and paper trading, so these describe the pipeline, not a
production service. They become hard requirements only in Phase 2.

### Pipeline SLIs/SLOs

| SLI | How measured | SLO |
|---|---|---|
| Brain throughput | Wall-clock seconds per 1,000 decisions | ≤ 150 s on 8 cores |
| Full experiment turnaround | One four-arm walk-forward run | ≤ 12 h unattended |
| Reproducibility | Re-run from seed, compare metrics | Bitwise for brain output; ≤ 1e-6 drift on metrics |
| Data completeness | Missing bars vs exchange calendar | ≤ 0.5% per month, all gaps logged |
| Data correctness | Cross-source spot check on random days | ≤ 2 bp median close difference |
| Leakage checks | Automated assertions per run | 100% pass, run blocks on failure |

### News channel SLIs/SLOs

| SLI | How measured | SLO |
|---|---|---|
| Jev latency p50 / p95 | Client-side per call | ≤ 400 ms / ≤ 1.5 s |
| Jev error rate | Failed calls after retries | ≤ 0.5% |
| Point-in-time integrity | Headline timestamp vs decision timestamp | 100% strictly before |
| Cost per experiment | Tokens × price | ≤ $5 per full run |

### Live paper-trading SLIs/SLOs (Phase 1)

| SLI | How measured | SLO |
|---|---|---|
| Decision latency | Bar close → order intent | p95 ≤ 5 s on a 15-minute grid |
| Session uptime | Minutes with a live data feed | ≥ 99% of market hours |
| Stale data guard | Decisions made on data older than 90 s | 0, hard block |
| Risk limit breaches | Position or loss limits exceeded | 0, kill switch trips first |
| Kill switch response | Command to flat | ≤ 10 s |
| Unexplained orders | Orders without a logged decision record | 0 |

Every SLO breach is logged with cause. Repeated breaches of the risk or
integrity SLOs block phase advancement regardless of returns.

## 3. Phase gates

Money only moves when every gate for the phase passes. No partial credit.

### Gate 0 → 1 (backtest to live paper)

1. Wiring advantage measured with CI, positive or negative, protocol unchanged
   since it was written down.
2. Leakage assertions pass on every fold.
3. Net Sharpe after costs beats both buy-and-hold and always-flat, **and**
   beats the rewired twin.
4. Sign survives 2× cost sensitivity.
5. Replication on a second market has the same sign.

### Gate 1 → 2 (live paper to real money, if ever)

1. ≥ 3 months of live paper trading with no risk or integrity SLO breach.
2. Live decision quality within one standard error of the backtest.
3. Simulated pass rate against funded-account rules (daily loss, total
   drawdown, consistency) reported over ≥ 200 simulated attempts.
4. Kill switch tested under a real failure.
5. A written pre-mortem: what would make this lose money, and what we would
   see first.

If Gate 0 fails, the project **still ships**: the negative result, the code and
the visualisation. That path is a success, not an abandonment.

## 4. Reporting

Every run writes a report containing: git commit, seed, data range, arm
configuration, all KPIs with CIs, SLI summary, leakage assertion results, and
wall-clock cost. Runs without a full report do not count as evidence.
