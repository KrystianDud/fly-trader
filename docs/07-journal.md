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

### The news pipeline, and what the live view exposed

Building the news channel properly meant three components, and the live monitor
found bugs in all of them within minutes of existing, which is the argument for
building the monitor.

**Volume forces a cascade.** The full history is about 5.35M relevant articles.
Jev scores 53 a second, so scoring everything is 28 hours — for $13. Cost is
irrelevant; throughput is the constraint. Asking Jev less per call does not
help either, because its questions run in parallel inside one call, so a
one-question triage pass costs the same wall-clock as a three-question scoring
pass. Only a local model changes the exponent.

So a classifier trained on Jev's own labels reads everything and forwards a few
percent. At a 5% keep rate it runs at perfect precision and at the ceiling of
achievable recall: every item it forwards is one Jev would have flagged. Full
history scoring drops from 28 hours to under three.

**What the monitor caught.** Within minutes of the three-column view existing:
the relevance filter was matching GDELT *themes* rather than headlines, so film
reviews and airline awards counted as market news — one slice went from 45
articles to 9 once relevance had to be visible in the headline. Syndicated
copies were arriving three and four times, sharing a headline across different
URLs. And the forwarded column was permanently empty because the page decided
what counted as forwarded using a threshold while the server forwarded the top
six by rank.

**A guess where a fact was available.** The loop assumed GDELT lags five
minutes and waited on that assumption, so a slice already published sat
untouched and the page looked dead for a quarter of an hour at a time. GDELT
publishes `lastupdate.txt` naming its current files. Ask, do not guess.

**Two channels, two jobs.** GDELT is the only source with archives, so it
remains the historical backfill, but its cadence is fifteen minutes and its
headlines are reconstructed from URL slugs. RSS is seconds-fresh with real
titles and has no past whatsoever. 28 feeds live, the widest net being Google
News queries that aggregate thousands of publishers continuously.

One deliberate choice worth recording: the RSS reader marks everything already
in the feeds as seen at startup without emitting it. Otherwise launching the
monitor dumps hundreds of hours-old stories in as though they had just broken,
which would look impressive and be a lie.

### The first real answer, and it is no

Four arms of six, full 3.7-year history, 92,030 decisions each, purged
walk-forward, costs included.

| Readout input | IC | Hit rate | Net bp | Sharpe |
|---|---|---|---|---|
| raw features only | **+0.0095** | 0.508 | -0.310 | -11.86 |
| real connectome | -0.0022 | 0.495 | -0.361 | -13.28 |
| rewire 0.1 | +0.0040 | 0.497 | -0.372 | -13.68 |
| rewire 0.25 | +0.0040 | 0.498 | -0.357 | -12.70 |
| rewire 0.5 | -0.0012 | 0.499 | -0.358 | -13.02 |

**The real wiring is not better than a scramble of itself.** No ordering by how
much structure was destroyed; the dose-response curve is flat. That was the
pre-registered primary question and the answer is no.

**The brain discards signal it was handed.** This is the part worth
understanding. The six raw market features carry an IC of +0.0095 on their own,
with signs that make economic sense — strong up-moves are followed by
down-moves, which is mean reversion. Pass those same features through 165,836
neurons and the information is gone. Not transformed, not concentrated
somewhere else in the population: gone. And scrambled wiring loses it equally,
which is why the arms are indistinguishable.

**Nothing pays for its own spread.** Even the best arm loses 0.31 bp per
decision, because an IC of 0.01 cannot cover 0.6 bp of cost paid on 40% of
92,030 decisions. Buy-and-hold returns Sharpe +0.48 and beats every model here.

#### The diagnostic that makes this a result rather than a bug report

A zero is only worth reporting if the apparatus can produce a non-zero. Planting
a noisy copy of the answer into the feature matrix gives IC +0.245, hit rate
63%, Sharpe +31. Labels, purged folds, fitting and evaluation all work. The
zeros are real zeros.

Second check: the target's own autocorrelation at 15-minute horizon is +0.002
to -0.004. USD/JPY at this frequency is close to a martingale, which is what
theory says it should be.

#### What the experiment was actually asking

Adding the features-only baseline changed the question. Until then the
comparison was "can the fly beat nothing", which a flat zero answers
ambiguously — an unpredictable market and a brain that destroys signal look
identical. With the baseline in place the question becomes "does the brain add
anything to the features it was given", and the answer is clearly negative: it
subtracts.

That baseline cost four lines and should have been there from the first run.

---

*The 50 ms ablation and the retinal arm are still pending. Neither is expected
to overturn this, but the 20 ms window does cut the network off while its
activity is still rising, and the retina is a genuinely different pathway.*
