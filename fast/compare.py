#!/usr/bin/env python3
"""Print the numbers that can actually be compared ACROSS arms, read from the live checkpoints.

    python3 fast/compare.py [--run ~/lisa/run] [--at STEP]

WHY THIS EXISTS.  The trainer's log line prints `val`, which is val_loss_fast()'s first element:
EVERY ARM'S OWN OBJECTIVE.  Those objectives are not the same function.  det_paper minimises a bare
waveform term (lambda = 0, LISA's released config); det adds 0.01 x a spectral term to it; the -erb
arms fold lambda x (ERB term) INSIDE the objective, so a bigger lambda mechanically inflates the
number.  Read as a leaderboard, that column ranks the arms by which loss function they were handed.
es_erb_l0.1's `val` is large because lambda is 0.1, not because the arm is losing.

val_wave is the same waveform term for every arm.  It is recorded in the checkpoint history and
never printed by the trainer, which is the gap this fills.

Arms reach a given step at different times, so the table is taken at the latest step EVERY arm has
validated at -- comparing arm A at step 40000 against arm B at step 20000 is not a comparison.

Checkpoints are written to a temp file and renamed, so a read here cannot catch a partial write.
Safe to run against a live run.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, STEPS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(pathlib.Path.home() / "lisa" / "run"))
    ap.add_argument("--at", type=int, default=None,
                    help="compare at this step (default: latest step every arm has validated at)")
    a = ap.parse_args()
    import torch

    run = pathlib.Path(a.run)
    hists, latest = {}, {}
    for arm in ARMS:
        ck = run / f"{arm}.pt"
        if not ck.exists():
            continue
        blob = torch.load(ck, map_location="cpu", weights_only=False)
        hists[arm] = blob["hist"]
        latest[arm] = blob["step"]
    if not hists:
        sys.exit(f"no checkpoints in {run} yet")
    missing = [a_ for a_ in ARMS if a_ not in hists]
    if missing:
        print(f"WARNING: no checkpoint for {', '.join(missing)} -- table is incomplete\n")

    sets = [set(h["val_step"]) for h in hists.values()]
    common = set.intersection(*sets) if sets else set()
    if not common:
        sys.exit("no validation step reached by every arm yet -- try again in a few minutes")
    step = a.at if a.at is not None else max(common)
    if step not in common:
        sys.exit(f"step {step} is not a validation step common to all {len(hists)} arms; "
                 f"nearest below is {max(c for c in common if c <= step) if any(c <= step for c in common) else 'none'}")

    rows = []
    for arm, h in hists.items():
        i = h["val_step"].index(step)
        kind, lam, cls = ARMS[arm]
        rows.append((h["val_wave"][i], h["val_spec"][i], h["val_loss"][i],
                     arm, kind, lam, cls, latest[arm]))
    rows.sort()

    print(f"all {len(rows)} arms at step {step}/{STEPS}  ({100 * step / STEPS:.1f}% of the run)\n")
    print(f"  {'arm':<18} {'val_wave':>10} {'val_spec':>10}  | {'own obj':>9}  "
          f"{'kind':<13} {'lambda':>6} {'model':<7} {'now at':>7}")
    print(f"  {'-' * 18} {'-' * 10} {'-' * 10}  + {'-' * 9}  {'-' * 13} {'-' * 6} {'-' * 7} {'-' * 7}")
    for w, s, own, arm, kind, lam, cls, now in rows:
        print(f"  {arm:<18} {w:10.6f} {s:10.6f}  | {own:9.6f}  {kind:<13} {lam:>6g} {cls:<7} {now:>7}")

    print("\n  val_wave is the one column comparable across every arm: same waveform term, same")
    print("  fixed validation batches.  Lower is better.  val_spec likewise.")
    print("\n  'own obj' is what the log prints.  It is comparable ONLY between arms sharing a")
    print("  (kind, lambda) pair, because only those minimise the same function:")
    groups = {}
    for arm, (kind, lam, _) in ARMS.items():
        groups.setdefault((kind, lam), []).append(arm)
    for (kind, lam), members in sorted(groups.items()):
        mark = "comparable" if len(members) > 1 else "alone -- nothing to compare it to"
        print(f"    {kind:<13} lambda={lam:<6g} {', '.join(members):<40} {mark}")


if __name__ == "__main__":
    main()
