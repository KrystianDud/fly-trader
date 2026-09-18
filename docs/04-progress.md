# Progress

Last updated: 2026-09-17

**Phase: 0 — research, no money at risk.**

## Summary

| Area | State |
|---|---|
| Brain loading and simulation | ✅ working, benchmarked |
| Market data | ✅ USD/JPY 1m, 1.38M bars |
| News sense (Jev) | ✅ connected and sanity-checked |
| News ingestion | ✅ GDELT (archival) + 28 RSS feeds (live) |
| Triage classifier | ✅ perfect precision at 5% keep |
| Live pipeline monitor | ✅ websocket, three stages |
| Sensory encoding | ✅ working, validated |
| Readout and training | ⬜ not started |
| Baseline arms | ⬜ not started |
| Backtest and validation | ⬜ not started |
| Risk layer | ⬜ not started |
| 3D visualisation | ⬜ not started |

Overall: roughly 40% of the way to a first end-to-end result. The go/no-go
test passed — market state reaches the descending neurons in a structured way.

## Done

### Brain
- MaleCNS v1.0 downloaded (annotations 14 MB, connectivity 1.0 GB,
  neurotransmitters 42 MB). CC-BY 4.0.
- Graph built: **165,836 neurons, 6,242,118 connections** at ≥ 5 synapses,
  signed by neurotransmitter. Matches the published headline figures
  (166k neurons, ~125M synapses).
- Leaky integrate-and-fire simulation implemented in PyTorch with the
  published parameters. Runs the whole brain in **0.125 s per 20 ms** of brain
  time on 8 CPU cores.
- Verified step-size invariance after fixing a bug where external drive was not
  scaled by the timestep, which silently suppressed all spiking at larger steps.
- Confirmed the input and output circuits exist in the data: 666 looming
  detectors, ~3,400 neurons per motion subtype, 2,639 olfactory neurons,
  2,558 tactile neurons, 2 Giant Fibers, 1,314 descending neurons.

### Data
- USD/JPY 1-minute bars, **1,380,998 rows**, 2023-01-01 → 2026-09-16, 22 MB.
- Availability surveyed: yen crosses and EUR/USD from 2013, Bund and Gilt from
  2018, US T-Bond from 2022, AAPL/MSFT from mid-2017, Nasdaq index only 2024.

### News sense
- `typesafe-sdk` installed, key wired through `.env`.
- Five-headline sanity check: four correct including a non-obvious FX inference
  (hot US inflation → weaker yen), one wrong in an instructive way (read
  "yen slides" literally, missing the implied intervention).
- Measured: **270 ms p50**, 636 ms worst case.

### Decisions recorded
- Six decisions logged in `05-decisions.md`, including the choice of USD/JPY
  over Nasdaq and the exclusion of numeric input to Jev.

### Go/no-go test passed (2026-09-17)

256 windows sampled across 2023-2026, whole brain, 20 ms each:

- Descending activity varies with market state: 209 spikes per window on
  average, range 0-415, with 358 of 1,314 descending neurons participating.
- Individual command neurons track individual signals, up to r = +0.97.
  Notably DNa02, a known steering neuron, tracks 15-minute momentum at +0.93.
- **Giant Fiber (DNp01) tracks the crash signal at r = +0.90**, rising
  monotonically across quartiles (3.8 -> 11.4 spikes). The escape circuit
  responds to sharp adverse moves.
- Five principal components explain 83% of descending variance; the top three
  map onto crash (+0.97), volatility (+0.95) and slow momentum (+0.85).

Caveat recorded: these correlations are partly by construction, since the
looming channel is driven by the crash signal. They show the signal propagates,
not that the wiring is better than a scrambled one. No relationship to future
returns has been tested yet.

## In progress

- Readout and confidence gate.

## Not started

Baseline arms, backtest harness, risk layer, news history ingestion,
visualisation. See `03-architecture.md` for the component list.

## Known issues

| Issue | Impact | Plan |
|---|---|---|
| Batching across market windows gave only ~1.4x | Full-history runs slower than hoped | Multiprocessing across 8 cores instead |
| 0.49 s per window with all channels driven, vs 0.125 s benchmark | 92k windows would take 12 h per arm | Multiprocessing, or train on a subsample first |
| 956 of 1,314 descending neurons never fire | Most of the command channel unused | Expected with only three sense modalities driven; revisit if the readout is starved |
| No GPU acceleration available | CPU-bound throughput | Accept; reduce decisions or graph density if binding |
| Point-in-time headlines not yet sourced | News channel cannot be backtested | GDELT ingestion, or forward-only testing |
| Dukascopy data unverified against a second source | Data-correctness SLO unmet | Spot-check before Gate 0 |

## Immediate next steps

1. Build the readout and confidence gate.
2. Build the rewired and random arms.
3. Build the walk-forward harness with costs and leakage assertions.
4. First four-arm result on USD/JPY.

## Log

| Date | Entry |
|---|---|
| 2026-09-17 | Go/no-go passed: market state reaches descending neurons, Giant Fiber tracks crashes at r=+0.90. |
| 2026-09-17 | Project started. Brain loading, simulation, benchmarking, USD/JPY data, Jev connection, documentation. |
