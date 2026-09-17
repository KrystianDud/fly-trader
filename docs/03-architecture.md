# Architecture

Last reviewed: 2026-09-17

## Data flow

```
market bars (1m)  ──►  features  ──┐
                                   ├──►  sensory drive  ──►  FIXED BRAIN  ──►  descending
headlines  ──►  Jev  ──►  scalars ─┘     (named cell       (LIF, 165,836      neuron
                                          populations)       neurons)          spike counts
                                                                                    │
                                                                                    ▼
                                                                          small trained readout
                                                                                    │
                                                                                    ▼
                                                                      long / flat / short + confidence
                                                                                    │
                                                                                    ▼
                                                                    risk limits ──► paper execution
```

The only trained component is the readout. Everything upstream is fixed by
biology or by a design decision recorded in `05-decisions.md`.

## Components

| Module | Responsibility | Status |
|---|---|---|
| `fly_trader/connectome.py` | Load MaleCNS into a signed sparse matrix; select neurons by annotation | done |
| `fly_trader/lif.py` | Leaky integrate-and-fire simulation, batched | done |
| `fly_trader/senses.py` | Market features → drive on named cell populations | drafted |
| `fly_trader/readout.py` | Descending spikes → decision + confidence | not started |
| `fly_trader/arms.py` | Rewire, random graph and conventional baselines | not started |
| `fly_trader/backtest.py` | Walk-forward, costs, slippage, leakage assertions | not started |
| `fly_trader/news.py` | Jev client, point-in-time headline handling, caching | not started |
| `fly_trader/risk.py` | Position and loss limits, kill switch, decision log | not started |
| `fly_trader/viz/` | 3D embodied fly reacting to market state | not started |

## The brain

- **Source**: MaleCNS v1.0, `minconf 0.5`, CC-BY 4.0.
- **Graph**: 165,836 annotated neurons, 6,242,118 connections at ≥ 5 synapses
  (124M synapses total in the unfiltered data).
- **Signs**: acetylcholine excitatory; GABA, glutamate and histamine
  inhibitory; modulators treated as weakly excitatory. Taken from the
  consensus neurotransmitter prediction per neuron.
- **Neuron model**: leaky integrate-and-fire with an alpha synapse, parameters
  from Shiu et al. 2024 (*Nature* 634:210): resting −52 mV, reset −52 mV,
  threshold −45 mV, membrane τ 20 ms, synaptic τ 5 ms, refractory 2.2 ms,
  synaptic delay 1.8 ms, 0.275 mV per synapse.
- **Integration**: explicit Euler. Verified invariant across dt ∈ {0.1, 0.2,
  0.5} ms once external drive is expressed as a rate in mV/ms.
- **Performance**: ~0.125 s per 20 ms of brain time, all 165,836 neurons, on 8
  CPU cores. PyTorch has no sparse support on Apple's GPU, so CPU only.

## Sensory mapping

Each market signal drives a real, identified cell type. See `senses.py` for the
table and `05-decisions.md` for why each was chosen.

Input channels: fast momentum (T4c/T5c, T4d/T5d), slow momentum (T4a/T5a,
T4b/T5b), volatility (tactile mechanosensory), adverse spike (LC4, LPLC2),
news (ORN_DA2 aversive, ORN_DM1 attractive).

Readout: the 1,314 descending neurons — the fly's own command channel from
brain to body.

## Experimental arms

| Arm | Graph | Purpose |
|---|---|---|
| `real` | MaleCNS as loaded | The hypothesis |
| `rewired` | Degree-preserving edge shuffle, weights and signs preserved | Controls for size, degree distribution and weight distribution. **The one that matters** |
| `random` | Erdős–Rényi, matched node and edge count | Controls for "any big sparse graph" |
| `mlp` | Conventional network, parameter count matched to the readout plus graph | Controls for "you just need a model" |

## Data

| Dataset | Source | Coverage | Location |
|---|---|---|---|
| Connectome | `storage.googleapis.com/flyem-male-cns/v1.0/...` | v1.0 | `data/connectome/` |
| USD/JPY 1m | Dukascopy | 2023-01 → present, 1,380,998 bars | `data/market/usdjpy_1m.parquet` |
| Yen crosses, bonds, AAPL/MSFT | Dukascopy | varies, see decisions log | not yet pulled |
| Headlines | GDELT (planned) | point-in-time | not yet pulled |

`data/` is git-ignored. Download commands live in `scripts/`.

## External services

| Service | Use | Constraints |
|---|---|---|
| **Jev** (`typesafe-sdk`) | Headline → threat/severity/kind scalars | 270 ms p50 measured; $0.042 per 1M input tokens; 64k token input cap; 1,200 req/min; key in `.env` |
| Dukascopy | Free historical bars | Rate-limited; no auth |

No broker integration exists and none is planned before Gate 1.

## Conventions

- Run everything with `PYTHONPATH=. uv run python ...`.
- Seeds fixed and recorded in every run report.
- All timestamps UTC internally; display in market-local time only.
- Secrets in `.env`, never in code, never in the repository.
