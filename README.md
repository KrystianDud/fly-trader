# fly-trader

A fruit fly's brain, wired to a market.

This project loads the real *Drosophila melanogaster* connectome (MaleCNS v1.0,
released June 2026), runs it as a spiking neural network with **fixed, untrained
weights**, and asks whether 166,000 neurons of evolved wiring make better
trading decisions than a scrambled copy of themselves.

The honest question is not "did the fly make money". It is:

> Does the real wiring beat a degree-preserving rewire of itself, under an
> identical training budget, after costs?

Everything in this repository is built to answer that question and to make a
negative answer just as publishable as a positive one.

## Documentation

| Document | Contents |
|---|---|
| [docs/01-scope.md](docs/01-scope.md) | Goals, non-goals, in scope, out of scope |
| [docs/02-metrics.md](docs/02-metrics.md) | KPIs, SLIs, SLOs, phase gates |
| [docs/03-architecture.md](docs/03-architecture.md) | Components and data flow |
| [docs/04-progress.md](docs/04-progress.md) | Current status and milestone log |
| [docs/05-decisions.md](docs/05-decisions.md) | Decision log with rationale |
| [docs/06-gpu-runbook.md](docs/06-gpu-runbook.md) | Running the sweep on a GPU box |
| [docs/07-journal.md](docs/07-journal.md) | What we found, including what was wrong |

## Quick start

```bash
uv sync
PYTHONPATH=. uv run python scripts/bench.py     # load the brain, time it
PYTHONPATH=. uv run python scripts/test_jev.py  # news sense check (needs .env)
```

Data is not in the repository. See [docs/03-architecture.md](docs/03-architecture.md)
for download commands.

## Status

Phase 0 (research, no money). See [docs/04-progress.md](docs/04-progress.md).

## Credits and licence

Connectome data: MaleCNS v1.0, Janelia Research Campus / Google Research,
CC-BY 4.0. Neuron model parameters: Shiu et al. 2024, *Nature* 634:210.
