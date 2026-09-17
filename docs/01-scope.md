# Scope

Last reviewed: 2026-09-17

## Goal

Determine whether the fixed wiring of a real fly brain carries useful inductive
bias for financial decision-making, measured against fair baselines, and
produce a result that stands up whether the answer is yes or no.

### Success looks like

A reproducible experiment that reports, with confidence intervals, the
difference in performance between the real connectome and a degree-preserving
rewire of itself on the same task, same data, same training budget, after
realistic costs. Plus a runnable demo that makes the idea legible to people
who will never read the numbers.

Explicitly: **a clean negative result counts as success.** "A fly brain is no
better than random wiring at trading" is a finding, and publishing it is
cheaper and more useful than quietly shelving the project.

## Primary objectives

1. **Scientific** — measure real wiring vs rewired vs random graph vs matched
   conventional model, under identical budgets.
2. **Engineering** — a pipeline that runs end to end on one laptop, is
   reproducible from a seed, and is honest about costs and leakage.
3. **Communication** — a 3D embodied visualisation that shows the fly reacting
   to market conditions in a way a non-specialist can follow.

## Secondary objectives

- Test whether a semantic news channel (Jev / TypeSafe System One) adds signal
  on top of price-derived channels.
- Test whether multi-timeframe input (1-minute and 15-minute) is merged usefully
  by the real wiring, where a scrambled graph would have to learn the merge.
- Establish whether the result transfers to a second, independent market.

## In scope

| Area | Included |
|---|---|
| Brain | MaleCNS v1.0, whole-CNS graph, leaky integrate-and-fire, fixed weights |
| Training | A small readout only (descending-neuron activity → decision) |
| Markets | USD/JPY primary; bonds, yen crosses, AAPL/MSFT as secondary tests |
| Timeframes | 1-minute input, 15-minute decisions |
| News | Timestamped historical headlines through Jev, point-in-time only |
| Validation | Walk-forward, transaction costs, slippage, regime splits |
| Execution | Paper/simulated only in Phase 0 and 1 |
| Hardware | One MacBook M3, CPU only |

## Out of scope

| Excluded | Why |
|---|---|
| **Real money in Phase 0 or 1** | Gated behind the metrics in `02-metrics.md`. No exceptions. |
| Training the connectome weights | Would destroy the attribution: we could no longer say the wiring did anything |
| High-frequency or sub-minute trading | Brain simulation takes ~0.125 s per decision; news adds 270 ms |
| Leverage, options, shorting complex instruments | Risk not justified by an experiment |
| Feeding numeric series to Jev | Documented weakness: not a calculator, weak numerical calibration |
| Claiming biological realism | Our neuron model is a simplification with no plasticity, no neuromodulation |
| Claiming the fly "thinks" or "decides" | It is a fixed filter with a trained readout. Say so, always |
| Flapping-wing flight simulation | Needs GPU cluster; a rigid-body flying visualisation is the substitute |
| GPU or cloud training | Revisit only if laptop throughput becomes the binding constraint |

## Non-goals worth stating plainly

- This is **not** a product, a fund, or a signal service.
- This is **not** an attempt to beat professional quantitative traders.
- The connectome is **not** expected to have an edge. The reservoir literature
  suggests at most better resistance to overfitting, which is a modest claim.

## Assumptions

- Free Dukascopy data is accurate enough for research. To be spot-checked
  against a second source before any gate decision.
- Historical headlines can be obtained point-in-time (GDELT). If not, the news
  channel is tested forward-only from the day it is switched on.
- CC-BY 4.0 permits our use of the connectome with attribution. It does.

## Risks

| Risk | Mitigation |
|---|---|
| Lookahead leakage via features or news | Walk-forward with embargo; point-in-time news only |
| Overfitting the four-way comparison itself | Fix the protocol before running; one shot per configuration |
| Result is a fluke of one market | Replication on a second market is a gate, not an extra |
| Laptop too slow for the full history | Decisions on a 15-minute grid; multiprocessing; subsample if needed |
| Enthusiasm outrunning evidence | Every public claim must name what was trained and what was fixed |
