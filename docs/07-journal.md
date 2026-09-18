# Journal

An honest log of what we found, including the things that were wrong. Written
as we go, so the dead ends stay visible rather than being tidied out of the
story afterwards.

---

## Day 1 — 2026-09-17

### The whole brain runs on a laptop

Expected to need a cluster. The MaleCNS connectome is 165,836 neurons and 6.24M
connections at a 5-synapse threshold, and a leaky integrate-and-fire simulation
of all of it runs at 0.125 s per 20 ms of brain time on eight CPU cores. The
published headline figures (166k neurons, ~125M synapses) reproduce exactly
from the raw files, which was a good sign the loader was right.

Apple's GPU turned out to be useless here: PyTorch has no sparse support on
Metal. CPU it was.

### A bug that would have been invisible

External drive was not being scaled by the integration step, so at larger
timesteps the input silently vanished and the network fell completely silent —
zero spikes, no error, no warning. Caught only because spike counts were
compared across step sizes. Anything that produces a plausible-looking zero is
the dangerous kind of bug.

### The fly's escape reflex tracks market crashes

The first real result. Feeding market state into named cell types and reading
the 1,314 descending neurons — the fly's own command channel to its body:

- Giant Fiber (DNp01), the escape neuron, correlates **+0.90** with sharp
  adverse price moves, rising monotonically across quartiles.
- DNa02, a known steering neuron, tracks 15-minute momentum at **+0.93**.
- Five components explain 83% of descending variance, and the top three map
  onto crash, volatility and slow momentum.

Caveat recorded at the time and still true: the looming channel is *driven* by
the crash signal, so some of this is by construction. It shows the signal
propagates, not that the wiring is clever.

### Jev reads the market better than a lexicon

Testing the news channel on five headlines, four were right — including one
that required real inference: hot US inflation implies a *weaker* yen, which it
got without being told how currencies work.

The fifth was wrong in an instructive way. "Emergency BOJ meeting called after
yen slides past 165" was read as yen-negative, because it took the words
literally and missed that an emergency meeting implies intervention. That
matches the model's documented weakness on indirection, and it set the rule we
have used since: **ask what the text says, never what it implies**.

---

## Day 2 — 2026-09-18

### The rewiring bug the standard deviations caught

The control graphs are supposed to preserve every neuron's degree while
destroying which neuron connects to which. The first implementation looked
fine until the summary printed out-degree and in-degree spreads that were
*swapped* relative to the real graph. The matrix is stored as `W[post, pre]`,
and the rewire had treated row as source, transposing the whole graph.

A silently transposed connectome would have invalidated every comparison in the
project, and nothing else would have flagged it.

### A dose-response knob that works

After the fix, rewiring a fraction of edges preserves out-degree, in-degree and
the inhibitory share exactly, while reciprocity — real structure — decays
smoothly:

| Arm | Reciprocity |
|---|---|
| real | 0.157 |
| rewire 0.1 | 0.127 |
| rewire 0.25 | 0.089 |
| rewire 0.5 | 0.041 |
| rewire 1.0 | 0.002 |
| random graph | 0.000 |

That turns the headline experiment from a binary comparison, which can always
be luck, into a curve.

### Validation that fails a Sharpe of 1.86

Sanity-checking the deflated Sharpe on pure random noise: a strategy scoring
1.86 comes back with a deflated probability of 0.51 against a luck benchmark of
1.74, because twenty configurations were tried. Working as intended. Most
retail backtests would have reported the 1.86.

### We were reading the fly mid-thought

Time-binned spike counts across 30,000 windows showed the distribution across
the four 5 ms bins was 0.000 / 0.093 / 0.358 / 0.549. Two things follow: the
first bin can never contain anything, because synaptic delay and membrane
charging take longer than 5 ms; and **more than half the spikes land in the
final bin**, meaning activity is still climbing when the simulation is cut off.

20 ms is probably too short. Added a 50 ms ablation rather than changing the
pre-registered protocol mid-flight.

### GPU: 18x, and a finding that evaporated

An L4 (the g5 with its A10G had no capacity anywhere in London) runs 71.6
windows/sec against the Mac's 3.94. The real arm over the full 3.7-year history
took 18.8 minutes.

Then the first rewired arm appeared to take over three times longer on
identical hardware, and a tidy explanation suggested itself: biological wiring
is clustered, so the sparse matrix has locality the GPU can coalesce, while
rewiring scatters connections and turns every multiply into a random fetch.
Evolved connectivity would be not only functionally organised but cheaper to
compute. A lovely side-finding.

It was wrong twice over.

Measuring the matrix multiply directly: rewiring costs **1.06x**, not 3x. And
the random graph, whose locality is thirteen times worse by any column-distance
measure, runs **faster** than the real connectome. Locality is not what drives
cost here.

Worse, the slowdown being explained was not real either. The instance logs in
UTC and the laptop runs on BST, so a one-hour clock offset was read as a
four-fold slowdown. The arm was running exactly on pace the whole time.

Two lessons, both cheap here and expensive elsewhere. A plausible mechanism is
not evidence — the locality story was convincing enough to write down before
anyone measured it. And before explaining a surprising number, check that the
number is real.

### The news model disagrees with the lexicon, correctly

Ingesting real GDELT headlines from the August 2024 yen carry unwind, GDELT's
own lexicon tone and Jev's reading correlate at **-0.39**. They disagree, and
Jev is right:

| Headline | Lexicon tone | Jev on the yen |
|---|---|---|
| japan stocks plunge leading global market sell off | -6.51 | 2.35, yen stronger |
| japans benchmark nikkei tumbles worst single day | -6.82 | 2.70, yen stronger |

A word-counting model sees catastrophe. A model that understands risk-off sees
yen demand. On 5 August 2024 the yen strengthened sharply. This is the clearest
evidence yet for the only defensible edge at this horizon: not reacting faster,
but reading second-order implication that faster systems get wrong.

### Showing the chart to the eye

The scalar encoding leaves 89,403 optic-lobe neurons — the largest structure in
the brain — completely idle. The dataset carries hexagonal column coordinates
for 23,720 of them, which is a retinotopic map, so the price chart can be drawn
onto a 36x39 lattice and played as a moving image during the simulation.

The correspondence with classical chart patterns is almost too neat: broadening
wedges are expanding objects, which is what LPLC2 looming detectors exist for;
trend lines are directional motion, which is T4 and T5. No pattern is defined
anywhere in the code. The fly gets pixels, and the rewired arms say whether
evolved shape-processing extracts more from them than a scramble does.

---

*Results from the six-arm sweep go here when it finishes.*
