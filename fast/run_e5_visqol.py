"""Run overnight3/e5_visqol.py against the converted OV50 checkpoints, on this laptop.

    python fast/run_e5_visqol.py --src ~/lisa-results

This is fast/run_e4_eval.py's job for the perceptual metrics: supply `test_utts`, point CKPT at the
converted checkpoints, suppress e5_visqol's own main block and call run_visqol from here.  It also
points the cell's `_find_eval_py()` at the repo, since evaluation.py lives beside the notebook
rather than in the results tree.

INSTALL.  Separate from the rest of the evaluation and not needed by it:

    pip install "visqol-python[lattice]" pesq torchmetrics torchaudio

visqol-python 3.8 is a pure-Python port of google/visqol -- no Bazel, no C++ toolchain, a
py3-none-any wheel.  The [lattice] extra pulls ai-edge-litert, which HAS a cp311 macosx_12_0_arm64
wheel, and that matters: without it e5_visqol falls back from the lattice speech-MOS mapping to the
polynomial one and the speech numbers stop being comparable with OV2.  The cell prints which mapping
is live and records it in the JSON; check that line before quoting a MOS.

WHAT THE TWO MODES SEE.  Speech mode resamples to 16 kHz, so its Nyquist is 8 kHz while the band the
model has to invent is 6-24 kHz: it sees about 11% of the problem, and it is reported only for
continuity with OV2 and the literature.  Audio mode runs at 48 kHz over 32 ERB bands to 24 kHz and
sees all of it -- lead with that.  NSIM is the mapping-free similarity and is the safer column in
both modes; MOS-LQO saturates around 4.7-4.75 rather than 5.0, so the `ceiling` condition is the
reference for what "perfect" scores here, not 5.

COST.  About 42 conditions per utterance (naive, one draw / mean / logmean per stochastic arm, a
passthrough variant of each, floor and ceiling), each scored by speech ViSQOL, audio ViSQOL and
PESQ.  --n-utts trims EVAL12 if you only want a smoke test; the JSON is rewritten after every
utterance, so an interrupted run still leaves usable partial results.
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import RUN_TAG
from fast.run_e4_eval import EVAL_CELLS
from fast.train_arm import boot

REPO = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--n-utts", type=int, default=12)
    ap.add_argument("--per-speaker", type=int, default=40)
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--table-only", action="store_true",
                    help="rebuild visqol_table_<tag>.md from visqol_<tag>.json without recomputing; "
                         "the ViSQOL pass is 29 minutes and re-rendering should not be")
    a = ap.parse_args()

    src = pathlib.Path(a.src).expanduser().resolve()
    ck_dir = src / "ckpt" / a.tag
    if not ck_dir.is_dir() or not list(ck_dir.glob("*.pt")):
        raise SystemExit(f"no checkpoints in {ck_dir} -- run fast/convert_ckpt.py --src {src} first")

    cwd = pathlib.Path.cwd()
    os.chdir(src)
    try:
        G = boot(device="cpu", root=src, cells=EVAL_CELLS)
    finally:
        os.chdir(cwd)
    import torch
    torch.set_num_threads(a.threads)
    if G["CFG"].name != "FULL":
        raise SystemExit(f"CFG is {G['CFG'].name}; the evaluation basis and rates must be FULL")

    from fast.vctk_local import test_utts
    utts, _ = test_utts(per_speaker=a.per_speaker, cache=src / "vctk_test")
    G["test_utts"] = utts
    G["OV3"] = str(REPO)                     # where e5_visqol's _find_eval_py() looks for evaluation.py
    G["OV3_TAG"], G["OV3_M"], G["OV3_RUN_EVAL"] = a.tag, a.M, False
    exec(compile((REPO / "overnight3/e5_visqol.py").read_text(), "<e5_visqol.py>", "exec"), G)

    models, _ = G["load_ov3"](a.tag) if "load_ov3" in G else (None, None)
    if models is None:                       # e4_eval was not exec'd in this process
        models = {}
        for p in sorted(ck_dir.glob("*.pt")):
            models[p.stem], _ = G["load_arm"](p)
            models[p.stem].eval()
    print(f"\n{len(models)} arms, M={a.M}, speech mapping {G['VISQOL_SP_MAPPING']}", flush=True)

    if a.table_only:
        d = json.loads((src / "ov3" / f"visqol_{a.tag}.json").read_text())
        TABLE = G["visqol_table"](d["agg"], d["n"], d["M"], d["speech_mapping"])
        (src / "ov3" / f"visqol_table_{a.tag}.md").write_text(TABLE)
        print(TABLE)
        print(f"\nrebuilt {src / 'ov3' / f'visqol_table_{a.tag}.md'} from the JSON", flush=True)
        return 0

    t0 = time.time()
    AGG, TABLE = G["run_visqol"](models, utts[:a.n_utts], a.M, a.tag)
    out = src / "ov3"
    print(f"\n[{time.time() - t0:.0f}s]  {out / f'visqol_{a.tag}.json'}", flush=True)
    print(f"          {out / f'visqol_table_{a.tag}.md'}", flush=True)
    print(f"\nspeech16k saw 8 kHz of a 24 kHz problem; lead with audio48k. "
          f"MOS-LQO saturates below 5 -- read the ceiling row for what perfect scores.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
