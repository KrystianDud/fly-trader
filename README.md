# fly-trader

A fruit fly's connectome, wired to a market.

This loads the real *Drosophila melanogaster* connectome (MaleCNS v1.0, released
June 2026), runs it as a spiking network with **fixed, untrained weights**, and
asks whether 166,000 neurons of evolved wiring make better trading decisions
than a scrambled copy of themselves.

The answer is no. More usefully: the *reason* turned out to be a problem with
the simulation rather than with connectomes, and that only became visible
because of a control that should have been run first.

## Findings

**1. No wiring advantage.** Across three encodings — market state as scalars at
20 ms and 50 ms, and the price chart drawn on the fly's retina as a moving
image — the real connectome never beat a degree-matched scramble of itself.
92,030 decisions per arm, 3.7 years of USD/JPY, purged walk-forward validation,
realistic costs.

| Arm | Information coefficient |
|---|---|
| raw market features (no brain) | **+0.0095** |
| random graph | +0.0083 |
| rewired connectome | +0.0031 |
| **real connectome** | **−0.0022** |

The dose-response curve — rewiring 10%, 25%, 50% and 100% of edges — is flat.

**2. The apparatus was not blind.** A classifier distinguishes the real
connectome from its degree-matched scramble with **99.2% accuracy** from
descending-neuron activity patterns. The wiring leaves an enormous signature.
That signature simply has nothing to do with prices.

**3. The real connectome compresses.** It collapses activity into ~7 effective
dimensions where a scramble uses 17–21, and the compression tracks the rewiring
dose monotonically. That is what a nervous system is for — funnelling
high-dimensional sensory input into a few mutually exclusive commands — and it
is precisely wrong for extracting weak statistical evidence.

**4. And then the control failed.** Asked to discriminate looming from receding
objects — the fly's best-characterised reflex, mediated by LC4 and LPLC2 feeding
the Giant Fiber — **the scrambled connectome outperformed the real one** (0.975
versus 0.920, with feature budgets matched).

So the honest scope of finding 1 is narrower than it first appeared:

> *This* leaky integrate-and-fire implementation — uniform neuron parameters,
> chemical synapses only — cannot reproduce the fly's own escape reflex. Until
> that is fixed, its silence on markets is evidence about the implementation,
> not about connectomes.

Two obvious suspects: every one of 165,836 neurons was given identical time
constants and thresholds, and the connectome data contains **no gap junctions**,
which the Giant Fiber system depends on.

## What is here

| | |
|---|---|
| `fly_trader/connectome.py` | MaleCNS into a signed sparse matrix: 165,836 neurons, 6.2M connections |
| `fly_trader/lif.py` | Leaky integrate-and-fire in PyTorch, parameters from Shiu et al. 2024 |
| `fly_trader/senses.py` | Market state onto named cell types — T4/T5 motion, LC4/LPLC2 looming, ORN olfactory |
| `fly_trader/retina.py` | The price chart drawn on the retinotopic lattice and played as a movie |
| `fly_trader/arms.py` | Degree-preserving rewires and random graphs: the controls |
| `fly_trader/validation.py` | Purged walk-forward, sample weights, deflated Sharpe |
| `fly_trader/graphmodel.py` | The connectome as a trainable architecture rather than a frozen filter |
| `scripts/positive_control.py` | Looming discrimination — the test that should have been first |
| `viz/index.html` | A 3D room where the fly trades, with its retinal input rendered live |
| [`docs/07-journal.md`](docs/07-journal.md) | **The honest log, including two retracted findings** |

## Reproducing

```bash
uv sync
bash scripts/gpu_setup.sh          # fetches the connectome, ~1.1 GB, CC-BY 4.0, no account
PYTHONPATH=. uv run python scripts/bench.py              # load and time the brain
PYTHONPATH=. uv run python scripts/extract.py --arm real --windows 30000
PYTHONPATH=. uv run python scripts/analyse.py
PYTHONPATH=. uv run python scripts/positive_control.py   # the control that matters
```

Whole-brain simulation runs at ~0.125 s per 20 ms of brain time on eight CPU
cores; an L4 GPU is about 18× faster. The full six-arm sweep cost £6.54 of
rented GPU.

## Method notes

The things that made the difference between a result and a story:

- **Degree-preserving rewires.** Every control preserves each neuron's in- and
  out-degree, the weight distribution and the inhibitory share exactly, while
  destroying which neuron connects to which. Reciprocity falls 0.157 → 0.002.
- **A positive control for the readout.** Planting a noisy copy of the answer
  scores IC +0.245 and Sharpe +31, so a zero elsewhere is a real zero.
- **A positive control for the biology**, which is the one that failed.
- **Purged walk-forward with embargo**, because overlapping labels leak.
- **Deflated Sharpe.** On pure random data a Sharpe of 1.86 deflates to 0.51
  once the number of configurations tried is accounted for.

## Documentation

| Document | Contents |
|---|---|
| [docs/01-scope.md](docs/01-scope.md) | Goals, non-goals, in scope, out of scope |
| [docs/02-metrics.md](docs/02-metrics.md) | KPIs, SLIs, SLOs, phase gates |
| [docs/03-architecture.md](docs/03-architecture.md) | Components and data flow |
| [docs/04-progress.md](docs/04-progress.md) | Status and milestone log |
| [docs/05-decisions.md](docs/05-decisions.md) | Decision log with rationale |
| [docs/06-gpu-runbook.md](docs/06-gpu-runbook.md) | Running the sweep on a GPU box |
| [docs/07-journal.md](docs/07-journal.md) | What we found, including what was wrong |

## Credits

Connectome: MaleCNS v1.0, Janelia Research Campus and Google Research, CC-BY
4.0. Neuron model: Shiu et al. 2024, *Nature* 634:210. Market data: Dukascopy.

Built in a day with Claude Code. The journal records what it got wrong as well
as what it got right, which felt more useful than a clean story.
