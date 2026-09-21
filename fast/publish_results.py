"""Copy an evaluation's outputs out of the results tree and into the repo.

    python fast/publish_results.py --src ~/lisa-results --tag OV50

WHY THIS IS A SCRIPT AND NOT A `cp`.  The evaluation runs against `~/lisa-results/`, which is a
scratch directory on one laptop: it holds the corpus cache, the converted checkpoints, 48 MB of
audio and a 6.9 MB corpus index, none of which belong in git, next to the handful of files that are
the actual result and currently exist nowhere else.  A hand-rolled copy gets that split wrong in one
direction or the other -- and the direction that loses is the one where a laptop dies with the only
copy of a 50-epoch, eight-arm run on it.

So the split is written down here, checked in, and repeatable: re-running an evaluation and
re-running this puts the new numbers in the same place with the same neighbours.

WHAT TRAVELS.  Everything text- or figure-shaped that cannot be recomputed without the corpus and a
GPU: the per-arm training histories (the run's primary record, and the thing every claim in
notes/ is checked against), the trainer logs, the run manifest and calibration, the EVAL12 results
and table, the val_wave verification, the ViSQOL table, and the two figures.  About 3.8 MB.

WHAT DOES NOT, and why:

    <arm>.pt              9.9 MB of resume checkpoints, and 4.4 MB of converted ones.  Binary, and
                          a decision about repo hygiene that is the repository owner's, not this
                          script's.  --checkpoints adds them if that decision goes the other way.
                          They ARE currently single-copy: the instance is terminated.
    ov3/audio/            48 MB of wav.  Derivable from the checkpoints in about three minutes.
    train_index.json      6.9 MB, and a pure cache -- fast/val_wave_check.py rebuilds it from the
                          Hub, and its correctness is gated on manifest.json either way.
    vctk_test/, lisa_rtm_cache/, ckpt/
                          corpus and checkpoint caches.

The README written into the destination records the same split, so the directory explains itself to
someone who arrives at it through git rather than through this file.
"""
import argparse
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, RUN_TAG

REPO = pathlib.Path(__file__).resolve().parents[1]

# (source path relative to --src, what it is).  A missing file is reported, not fatal: the ViSQOL
# pass is a separate install and the val_wave check is a separate run, so a partial publish is a
# normal state rather than an error.
def plan(tag):
    items = [("manifest.json", "corpus fingerprint, g3 digests, test speakers"),
             ("calibration.json", "calibration probe from the run")]
    for arm in sorted(ARMS):
        items.append((f"history_{tag}_{arm}.json", f"{arm}: every logged train and val point"))
    for arm in sorted(ARMS):
        items.append((f"{arm}.log", f"{arm}: trainer log"))
    items += [(f"ov3/results_{tag}.json", "EVAL12: every metric, every arm, every condition"),
              (f"ov3/table_{tag}.md", "EVAL12 table"),
              (f"ov3/val_wave_{tag}.json", "val_wave verification: seeds, AMP, bands, residuals"),
              (f"ov3/visqol_{tag}.json", "ViSQOL / PESQ per condition"),
              (f"ov3/visqol_table_{tag}.md", "ViSQOL table"),
              (f"ov3/metrics_{tag}.md", "every metric, all eight arms, one document"),
              (f"figs/ov3_spectrum_{tag}.png", "energy ratio per band"),
              (f"figs/ov3_calibration_{tag}.png", "PIT histograms and spread-skill"),
              (f"figs/train_val_{tag}.png", "per-arm training and validation curves"),
              (f"figs/cross_arm_{tag}.png", "val_wave, val_spec and spread on shared axes"),
              (f"figs/endgame_{tag}.png", "the last 12k steps, linear axes")]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--dest", default=None, help="default overnight3/results_<TAG>")
    ap.add_argument("--checkpoints", action="store_true",
                    help="also copy the converted <arm>.pt (4.4 MB); see this file's docstring")
    a = ap.parse_args()

    src = pathlib.Path(a.src).expanduser().resolve()
    dest = pathlib.Path(a.dest).expanduser().resolve() if a.dest else REPO / "overnight3" / f"results_{a.tag}"
    dest.mkdir(parents=True, exist_ok=True)

    copied, missing, total = [], [], 0
    for rel, what in plan(a.tag):
        p = src / rel
        if not p.exists():
            missing.append((rel, what))
            continue
        out = dest / pathlib.Path(rel).name
        shutil.copy2(p, out)
        copied.append((out.name, what, out.stat().st_size))
        total += out.stat().st_size
    if a.checkpoints:
        for arm in sorted(ARMS):
            p = src / "ckpt" / a.tag / f"{arm}.pt"
            if p.exists():
                out = dest / "ckpt" / f"{arm}.pt"
                out.parent.mkdir(exist_ok=True)
                shutil.copy2(p, out)
                copied.append((f"ckpt/{out.name}", f"{arm}: converted checkpoint", out.stat().st_size))
                total += out.stat().st_size

    lines = [f"# {a.tag} results", "",
             f"Copied out of `{a.src}` by `fast/publish_results.py` on "
             f"{time.strftime('%Y-%m-%d')}. Read them with `notes/2026-09-21-eval-results.md`.", "",
             "| file | what |", "|---|---|"]
    lines += [f"| `{n}` | {w} |" for n, w, _ in copied]
    if missing:
        lines += ["", "Not produced yet:", ""] + [f"- `{pathlib.Path(r).name}` -- {w}" for r, w in missing]
    lines += ["", "## Not here", "",
              "- **`<arm>.pt`** -- the eight resume checkpoints (9.9 MB) and their converted forms",
              "  (4.4 MB). `fast/publish_results.py --checkpoints` copies the converted ones. The",
              "  resume checkpoints are currently single-copy: the instance is terminated.",
              "- **`ov3/audio/`** -- 48 MB of wav, rebuildable in about three minutes with",
              "  `fast/run_e4_eval.py`.",
              "- **`train_index.json`** -- 6.9 MB corpus index, a cache;",
              "  `fast/val_wave_check.py` rebuilds it and gates it against `manifest.json`.", "",
              "## Reproducing", "",
              "```bash", "python fast/convert_ckpt.py   --src ~/lisa-results",
              "python fast/run_e4_eval.py    --src ~/lisa-results",
              "python fast/val_wave_check.py --src ~/lisa-results --seeds 1234,1,2,3 --amp",
              "python fast/run_e5_visqol.py  --src ~/lisa-results   # needs a separate install",
              "python fast/publish_results.py --src ~/lisa-results", "```", ""]
    (dest / "README.md").write_text(chr(10).join(lines))

    print(f"{len(copied)} files -> {dest}  ({total / 1e6:.1f} MB)")
    for n, _, s in copied:
        print(f"  {s / 1024:>9.1f} KB  {n}")
    for rel, _ in missing:
        print(f"  {'--':>9}     {pathlib.Path(rel).name}  (not produced yet)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
