# Decision log

Each entry records what was decided, why, and what would overturn it.

---

## D1 — The wiring stays fixed; only a readout is trained

**Decision.** Synaptic weights come from the connectome and are never updated.
The only trained parameters map descending-neuron spike counts to a decision.

**Why.** It is the only way the result means anything. FlyGM trains per-neuron
parameters throughout the graph, so its gains could come from the structure or
from the training. Keeping the wiring frozen means any advantage over the
rewired twin is attributable to biology.

**Cost.** Almost certainly weaker raw performance than a trained network.
Accepted.

**Overturned if.** The fixed brain produces descending activity with no
relationship to market state at all, in which case the design is dead rather
than conservative.

---

## D2 — USD/JPY as the primary market, not Nasdaq

**Decision.** Primary instrument is USD/JPY 1-minute bars.

**Why.** Three reasons, in order of weight:
1. **No overnight gaps.** The brain runs in continuous time with 20 ms membrane
   dynamics. Equities gap overnight, which is a discontinuity the model has no
   sense organ for. Currency trades 24/5.
2. **Depth.** Free 1-minute data back to 2013, versus 2024 for the Nasdaq index
   CFD and mid-2017 for large-cap US stocks.
3. **Reactivity.** Current yen and bond conditions produce the sharp moves a
   reflex system should be tested on.

**Cost.** Currency is closer to pure noise than equities, so any edge is harder
to find. Accepted — we are testing wiring, not hunting alpha.

**Overturned if.** Data quality problems appear, or the result needs an
equities replication to be credible.

---

## D3 — Jev handles text only, never numbers

**Decision.** The Jev model receives headlines and returns semantic scalars. It
never receives price series, indicator values or numeric comparisons.

**Why.** Its own documentation states it is "not a calculator", "cannot
reliably judge if two numeric values are close", has weak numerical calibration
on scores, and treats dates as text rather than ordered quantities. Our own
test confirmed strong semantic judgement and one clear failure of second-order
inference.

**Practical rule.** Ask what the text *says*, never what it *implies*.

**Overturned if.** A later model version documents numeric competence.

---

## D4 — Market signals drive named cell types, not arbitrary neurons

**Decision.** Each input channel targets an identified population: motion
detectors for momentum, mechanosensory neurons for volatility, looming
detectors for adverse spikes, olfactory receptor neurons for news.

**Why.** An arbitrary mapping would make the connectome a generic sparse graph
and waste the one thing that makes it interesting. Driving the looming
detectors with a crash means the escape circuit is doing the job it evolved
for, which is both more defensible and more legible.

**Caveat.** The mapping is still a human design choice, and a fly does not
actually see prices. This is stated plainly wherever results are reported.

**Overturned if.** Ablation shows channel assignment makes no difference, which
would itself be worth reporting.

---

## D5 — Two timeframes into two populations

**Decision.** 1-minute momentum drives the vertical-motion subtypes (T4c/T5c,
T4d/T5d); 15-minute momentum drives the horizontal-motion subtypes (T4a/T5a,
T4b/T5b). The wiring merges them; we never specify how.

**Why.** The fly's motion detectors come in populations tuned to different
speeds, so multiple timeframes have a genuine biological analogue. It is also
the sharpest test available: merging two timescales is exactly the kind of
structured computation that evolved wiring might do better than a scrambled
graph.

**Overturned if.** Single-timeframe input performs identically, making the
extra channel dead weight.

---

## D6 — Decisions on a 15-minute grid

**Decision.** One decision per 15 minutes, using 1-minute data as input.

**Why.** Brain simulation costs ~0.125 s per decision and news adds 270 ms.
A 15-minute grid keeps a full run within the 12-hour turnaround objective and
stays well clear of any claim to high-frequency trading.

**Overturned if.** Throughput improves enough to justify a finer grid, or the
signal proves to live at a different horizon.

---

## D7 — Negative results ship

**Decision.** If the real connectome does not beat its rewired twin, the
project publishes that, with the code and the visualisation.

**Why.** The experiment is only honest if the answer is allowed to be no.
It also protects against the failure mode where a disappointing result
quietly turns into a search for a better-looking one.

**Overturned if.** Never.
