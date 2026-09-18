"""Can this pipeline express the fly's best-documented reflex at all?

Every market result today compares a real connectome against a degree-matched
scramble and finds no difference. That is only meaningful if the apparatus is
capable of finding a difference when one exists.

So: looming detection. An expanding dark shape means something is about to hit
the animal, and the circuit that detects it (LC4 and LPLC2 feeding the Giant
Fiber, DNp01) is among the best characterised in neuroscience. The fly should
be visibly better at this than a scramble of itself.

Same retina, same lamina entry, same descending readout, same tiny linear
readout as the market runs. Only the stimulus changes.

  real >> scrambled  -> the apparatus works, and the market null is about
                        markets
  real ~= scrambled  -> the pipeline cannot reproduce the animal's most famous
                        reflex, and every market result today is measuring our
                        own plumbing instead of biology
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from fly_trader import arms, connectome, retina
from fly_trader.lif import FlyBrain, LIFParams


def stimuli(eye: retina.Retina, n: int, seed: int = 0, difficulty: float = 1.0):
    """Three visual worlds the fly has real opinions about.

    looming    an expanding dark disc: something is coming, escape now
    receding   the same disc shrinking: it is leaving, ignore it
    translating a disc of fixed size drifting sideways: ordinary motion

    The first version of this was too easy: clean discs on an empty field are
    nearly linearly separable, every arm scored 0.99+, and the test could not
    tell the arms apart. Difficulty adds what a fly actually faces — sensor
    noise, background clutter, partial occlusion, low contrast and variable
    speed — so that tuning has to do some work.
    """
    rng = np.random.default_rng(seed)
    H, W, F = eye.height, eye.width, eye.frames
    movies = np.zeros((n, F, H, W), dtype=np.float32)
    labels = np.zeros(n, dtype=int)

    yy, xx = np.mgrid[0:H, 0:W]
    for i in range(n):
        kind = i % 3
        labels[i] = kind
        cy = rng.uniform(H * 0.25, H * 0.75)
        cx = rng.uniform(W * 0.25, W * 0.75)
        r0 = rng.uniform(1.5, 3.0)
        speed = rng.uniform(0.4, 1.6)
        contrast = 1.0 - 0.45 * difficulty * rng.random()
        drift = rng.uniform(-0.8, 0.8) * difficulty   # looming things also drift

        # static clutter: a few distractor blobs that persist across frames
        clutter = np.zeros((H, W), dtype=np.float32)
        for _ in range(int(6 * difficulty)):
            by, bx = rng.uniform(0, H), rng.uniform(0, W)
            br = rng.uniform(0.8, 2.0)
            clutter = np.maximum(
                clutter,
                (((xx - bx) ** 2 + (yy - by) ** 2 <= br ** 2) * rng.uniform(0.3, 0.8))
            )

        for f in range(F):
            if kind == 0:
                r = r0 + speed * f
                x, y = cx + drift * f, cy
            elif kind == 1:
                r = max(0.8, r0 + speed * (F - 1) - speed * f)
                x, y = cx + drift * f, cy
            else:
                r = r0 + speed * (F - 1) / 2
                x, y = cx + (f - F / 2) * 1.2, cy

            frame = ((xx - x) ** 2 + (yy - y) ** 2 <= r ** 2).astype(np.float32) * contrast
            frame = np.maximum(frame, clutter)
            if difficulty > 0:                      # occlude a vertical band
                band = int(rng.uniform(0, W))
                frame[:, band : band + max(1, int(3 * difficulty))] = 0.0
                frame += rng.normal(0, 0.12 * difficulty, frame.shape)
            movies[i, f] = np.clip(frame, 0, 1)
    return movies, labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["real", "rewire1.0", "random"])
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--sim-ms", type=float, default=50.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-features", type=int, default=150,
                    help="cap live channels per arm so feature count is not the variable")
    ap.add_argument("--difficulty", type=float, default=1.0,
                    help="0 = clean discs (too easy), 1 = noise, clutter, occlusion")
    args = ap.parse_args()

    c = connectome.load(min_weight=5)
    eye = retina.Retina(c, frames=8)
    dn_ids = c.ids_where(superclass="descending_neuron")
    dn = torch.tensor(c.idx(dn_ids))
    gf_cols = [i for i, b in enumerate(dn_ids) if b in set(c.ids_where(type="DNp01"))]

    movies, labels = stimuli(eye, args.n, args.seed, args.difficulty)
    print(f"{args.n} stimuli: looming / receding / translating, "
          f"{eye.width}x{eye.height} retina, {args.sim_ms:g} ms each\n")

    results = {}
    for arm in args.arms:
        W = arms.build(c.W, arm, seed=args.seed)
        brain = FlyBrain(W, LIFParams(dt=0.2), device=args.device)
        t0 = time.time()

        acts = []
        for i in range(0, len(movies), args.batch):
            chunk = movies[i : i + args.batch]
            drive = torch.stack([eye.drive(m) for m in chunk], dim=2)
            out = brain.run_movie(args.sim_ms, eye.all_idx, drive, dn, bins=2)
            acts.append(out.permute(2, 0, 1).reshape(len(chunk), -1).cpu().numpy())
        A = np.concatenate(acts)
        A = A[:, A.std(0) > 0] if (A.std(0) > 0).any() else A

        # Arms keep different numbers of neurons alive (the real connectome is
        # selective, a scramble floods), and a linear classifier does better
        # with more channels regardless of their quality. Cap every arm at the
        # same budget, chosen by variance, so the comparison is about wiring
        # rather than about channel count.
        if args.max_features and A.shape[1] > args.max_features:
            keep = np.argsort(-A.std(0))[: args.max_features]
            A = A[:, np.sort(keep)]

        # can a linear readout tell the three worlds apart?
        if A.shape[1] == 0:
            acc = looming_acc = gf_ratio = float("nan")
        else:
            acc = float(cross_val_score(
                LogisticRegression(max_iter=2000, C=0.05), A, labels, cv=4
            ).mean())
            pair = labels < 2  # looming vs receding only: the decisive contrast
            looming_acc = float(cross_val_score(
                LogisticRegression(max_iter=2000, C=0.05), A[pair], labels[pair], cv=4
            ).mean())

            # does the Giant Fiber itself prefer looming, as it must in a fly?
            full = np.concatenate(acts)
            gf = full.reshape(len(full), 2, -1)[:, :, gf_cols].sum(axis=(1, 2))
            gf_ratio = float(
                (gf[labels == 0].mean() + 1e-9) / (gf[labels == 1].mean() + 1e-9)
            )

        results[arm] = {"three_way_acc": acc, "looming_vs_receding": looming_acc,
                        "giant_fiber_looming_ratio": gf_ratio,
                        "spikes_per_stimulus": float(np.concatenate(acts).sum(1).mean()),
                        "live_features": int(A.shape[1]), "minutes": (time.time() - t0) / 60}
        print(f"{arm:12s} 3-way {acc:.3f}  looming-vs-receding {looming_acc:.3f}  "
              f"GF ratio {gf_ratio:5.2f}  ({A.shape[1]} features, "
              f"{(time.time() - t0) / 60:.1f} min)", flush=True)
        del brain, A

    real, rew = results.get("real"), results.get("rewire1.0")
    if real and rew:
        d = real["looming_vs_receding"] - rew["looming_vs_receding"]
        print(f"\nlooming discrimination, real minus scramble: {d:+.3f}")
        at_ceiling = min(real["looming_vs_receding"], rew["looming_vs_receding"]) > 0.97
        if at_ceiling:
            print("VERDICT: inconclusive — both arms at ceiling, the task is too easy "
                  "to separate them. Raise --difficulty.")
        elif d > 0.05:
            print("VERDICT: apparatus expresses biology — the market null is about markets")
        else:
            print("VERDICT: apparatus cannot reproduce the fly's own reflex — treat every "
                  "market result as untrustworthy")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/positive_control.json").write_text(json.dumps(results, indent=2))
    print("chance level is 0.333 three-way, 0.500 for the pair")


if __name__ == "__main__":
    main()
