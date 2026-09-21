"""Run overnight3/e4_eval.py against the converted OV50 checkpoints, on this laptop.

    python fast/run_e4_eval.py --src ~/lisa-results

WHAT THIS ADDS TO e4_eval.py.  e4_eval is a Colab cell: it expects a booted notebook around it, with
`test_utts` already in globals from overnight2/c0_boot.py, and its trailing main block runs the
moment the file is exec'd.  This supplies the three things that boot does not:

    test_utts   from fast/vctk_local.py -- a partial read of the held-out speakers, not 11.7 GB
    CKPT/tag    ~/lisa-results/ckpt/OV50, what fast/convert_ckpt.py writes
    control     OV3_RUN_EVAL = False suppresses the main block, so run_eval is called from here
                with `audio` and the utterance count as arguments rather than as edits

DEVICE.  CPU, not MPS.  The model is 87k parameters and the decoder is a 5-layer MLP at 48 kHz, which
is dispatch-bound: measured 0.08 s per 3.5 s utterance on CPU against 0.68 s on MPS.  MPS also draws
different noise for the same seed (torch.Generator is per-device), so switching backends would move
every stochastic number for reasons that have nothing to do with the arms.

READ THE TABLE WITH THE CONVENTIONS IN notes/2026-09-20-eval-handoff.md:

  * LSD IS NOT IN dB.  lsd_db (build_notebook.py:650) omits the factor of 10 in 10*log10, so the
    column is decades of power; true dB is 10x. Every prior number in this repo uses the same form,
    so it is comparable internally and must NOT be silently "fixed".
  * SNR is genuine dB, but snr_db floors the error energy at 1e-20, so a perfect reconstruction
    returns a finite ~230 dB. Anything in that band is a passthrough bug, not a result.
  * det_paper and det are deterministic: M = 1, CRPS degenerates to mean absolute error and there is
    no PIT. They are the reference line on those two columns, not entries in the ranking.
  * PIT end-bins ideal is 2/(M+1) = 0.1176 at M = 16. Above is under-dispersed. The ranks are per
    time-frequency bin and heavily correlated, so a binomial error bar on the bin count is far too
    tight to judge a small deviation with.
"""
import argparse
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, RUN_TAG
from fast.train_arm import boot

REPO = pathlib.Path(__file__).resolve().parents[1]
# The inference cell, cut before the training loop it ends with.  See boot()'s `cells` docstring.
EVAL_CELLS = (("def sample_batch(utts, cfg, rng):", "CKPT_PATH = "),)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--M", type=int, default=16, help="ensemble draws per utterance for the stochastic arms")
    ap.add_argument("--n-utts", type=int, default=12, help="EVAL12; the handoff's ideals table assumes 12")
    ap.add_argument("--per-speaker", type=int, default=40)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args()

    src = pathlib.Path(a.src).expanduser().resolve()
    ck_dir = src / "ckpt" / a.tag
    if not ck_dir.is_dir() or not list(ck_dir.glob("*.pt")):
        raise SystemExit(f"no checkpoints in {ck_dir} -- run fast/convert_ckpt.py --src {src} first")

    cwd = pathlib.Path.cwd()
    os.chdir(src)                      # the notebook's setup cell builds ./lisa_rtm_cache in the CWD
    try:
        G = boot(device=a.device, root=src, cells=EVAL_CELLS)
    finally:
        os.chdir(cwd)
    import torch
    torch.set_num_threads(a.threads)

    CFG = G["CFG"]
    if CFG.name != "FULL":
        raise SystemExit(f"CFG is {CFG.name}; the evaluation basis and rates must be FULL")
    print(f"\nCFG {CFG.name}  {CFG.fs_lo} -> {CFG.fs_hi} Hz  eval basis n_fft={CFG.eval_n_fft} "
          f"hop={CFG.eval_hop} k_cut={CFG.eval_k_cut}  device {G['DEVICE']}  threads {a.threads}", flush=True)

    from fast.vctk_local import test_utts
    print(f"\ntest utterances ({a.per_speaker}/speaker, EVAL{a.n_utts} is the first {a.n_utts}):", flush=True)
    utts, names = test_utts(per_speaker=a.per_speaker, cache=src / "vctk_test")
    G["test_utts"] = utts
    G["test_names"] = names

    G["OV3_TAG"], G["OV3_M"], G["OV3_RUN_EVAL"] = a.tag, a.M, False
    exec(compile((REPO / "overnight3/e4_eval.py").read_text(), "<e4_eval.py>", "exec"), G)

    models, arms = G["load_ov3"](a.tag)
    missing = set(ARMS) - set(models)
    if missing:
        print(f"WARNING: no checkpoint for {sorted(missing)}", flush=True)
    print(f"\n{len(models)} arms, M={a.M} for the stochastic ones:", flush=True)
    for k in sorted(models):
        kind, lam, cls = arms[k]
        print(f"  {k:<18} {kind:<12} lam={lam:<7g} {cls:<7} tau={models[k].tau:g}"
              f"{'   (deterministic: M=1, CRPS=MAE, no PIT)' if models[k].tau == 0 else ''}", flush=True)

    eval_utts = [u for u in utts[:a.n_utts]]
    print(f"\nEVAL{len(eval_utts)}: {sum(len(u) for u in eval_utts) / CFG.fs_hi:.1f} s of audio\n", flush=True)
    t0 = time.time()
    RES, TABLE = G["run_eval"](models, arms, eval_utts, a.M, a.tag, audio=not a.no_audio)
    out = src / "ov3"
    print(f"\n[{time.time() - t0:.0f}s]  results {out / f'results_{a.tag}.json'}", flush=True)
    print(f"          table   {out / f'table_{a.tag}.md'}", flush=True)
    print(f"          figures {G['FIGS'] / f'ov3_spectrum_{a.tag}.png'}, "
          f"{G['FIGS'] / f'ov3_calibration_{a.tag}.png'}", flush=True)
    print("\nLSD above is decades of power, not dB -- multiply by 10 for dB. "
          "See this file's docstring before quoting any of it.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
