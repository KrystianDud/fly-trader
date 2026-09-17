# Progress

Last updated: 2026-09-17

**Phase: 0 — research, no money at risk.**

## Summary

| Area | State |
|---|---|
| Brain loading and simulation | ✅ working, benchmarked |
| Market data | ✅ USD/JPY 1m, 1.38M bars |
| News sense (Jev) | ✅ connected and sanity-checked |
| Sensory encoding | 🟡 drafted, untested |
| Readout and training | ⬜ not started |
| Baseline arms | ⬜ not started |
| Backtest and validation | ⬜ not started |
| Risk layer | ⬜ not started |
| 3D visualisation | ⬜ not started |

Overall: roughly a third of the way to a first end-to-end result.

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

## In progress

- **Sensory encoding** (`senses.py`) — written, not yet run against real data.
  Next step is checking that descending-neuron activity actually varies with
  market state. If it does not, the whole approach needs rethinking, so this is
  the first real go/no-go moment.

## Not started

Readout, baseline arms, backtest harness, risk layer, news history ingestion,
visualisation. See `03-architecture.md` for the component list.

## Known issues

| Issue | Impact | Plan |
|---|---|---|
| Batching across market windows gave only ~1.4× | Full-history runs slower than hoped | Multiprocessing across 8 cores instead |
| No GPU acceleration available | CPU-bound throughput | Accept; reduce decisions or graph density if binding |
| Point-in-time headlines not yet sourced | News channel cannot be backtested | GDELT ingestion, or forward-only testing |
| Dukascopy data unverified against a second source | Data-correctness SLO unmet | Spot-check before Gate 0 |

## Immediate next steps

1. Run the sensory encoder over a few thousand USD/JPY windows and check that
   descending-neuron activity varies with market state (go/no-go).
2. Build the readout and confidence gate.
3. Build the rewired and random arms.
4. Build the walk-forward harness with costs and leakage assertions.
5. First four-arm result on USD/JPY.

## Log

| Date | Entry |
|---|---|
| 2026-09-17 | Project started. Brain loading, simulation, benchmarking, USD/JPY data, Jev connection, documentation. |
