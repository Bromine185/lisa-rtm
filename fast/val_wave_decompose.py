"""Decompose val_wave into the energy score's parts, on the run's own validation batches.

    python fast/val_wave_decompose.py --src ~/lisa-results --seeds 1234,1

WHY.  fast/compare.py and notes/metrics-reference.md call val_wave "the same waveform term for every
arm".  It is the same PROPER SCORE.  It is not the same reconstruction error.  ArmStack.losses
(overnight3/e2b_fast.py:334-383) computes it two ways:

    is_det:  l_w = |yh - y|                                   plain L1 of the one output
    else:    dw  = 0.5(|y-y1| + |y-y2|) - 0.5|y1-y2|          two-draw energy score

The energy score of a point mass IS its MAE, so the det branch is the degenerate case of the es
branch and the column is a legitimate proper-score comparison, exactly as CRPS is.  What it is NOT
is a comparison of waveform error: a sampler's val_wave is its mean single-draw L1 MINUS half its
ensemble spread, and the spread is of the same order as val_wave itself.  Read as an error, the
column credits the samplers for something a deterministic arm structurally cannot have.

THIS FILE MEASURES THE PARTS RATHER THAN INFERRING THEM.  The history's `spread` is the TRAINING
batch's, under anchor jitter; the validation-time spread is recorded nowhere.  So per arm, on the
same batches and the same eps stream val_loss_fast uses:

    val_wave      via stack.losses()                        the real code path
    d1, d2, d12   via stack.forward() on the SAME eps       independent arithmetic
    identities    dw == 0.5(d1 + d2) - 0.5 d12  and  spread == d12, at tensor tolerance
    L1_single     0.5(d1 + d2), the mean single-draw waveform L1

and it ties itself to the sweep: the losses()-path val_wave at seed 1234 must equal the fp32
seed-1234 value in ov3/val_wave_<tag>.json, which was itself matched to the recorded curve.  If that
anchor fails, nothing below it is about the run.
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.run_contract import ARMS, RUN_TAG
from fast.train_arm import boot
from fast.val_wave_check import build_index, corpus_from_index, decode, draw_val, make_vb

REPO = pathlib.Path(__file__).resolve().parents[1]


def decompose(G, stack, batches, seed, torch):
    """One arm, one eps stream: every term of the waveform score, measured two ways."""
    need = G["_needs"]([stack])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    keys = ("loss", "wave", "spec", "spread", "d1", "d2", "l1")
    acc = {k: 0.0 for k in keys}
    worst = {"dw": 0.0, "spread": 0.0}
    with torch.no_grad():
        for x, y in batches:
            tf = G["TargetFeats"](y, stack.R, *need)
            B = x.shape[0]
            if stack.is_det:
                loss, w, s, sp = stack.losses(x, y, tf, perturb=False)
                yh = stack.forward(x, None, None, False)[0]                       # (B, T)
                l1 = (yh - y).abs().mean()
                worst["dw"] = max(worst["dw"], abs(float(w[0]) - float(l1)))
                d1 = d2 = l1
                d12 = torch.zeros(())
            else:
                eps = stack.sample_eps(2 * B, x.shape[1], gen)                    # SAME draw as val_loss_fast
                loss, w, s, sp = stack.losses(x, y, tf, eps=eps, perturb=False)
                e, d = eps
                yh = stack.forward(x.repeat(2, 1), e, d, False)[0]               # (2B, T)
                y1, y2 = yh[:B], yh[B:]
                d1 = (y - y1).abs().mean(-1).mean()
                d2 = (y - y2).abs().mean(-1).mean()
                d12 = (y1 - y2).abs().mean(-1).mean()
                dw_direct = 0.5 * (d1 + d2) - 0.5 * d12
                worst["dw"] = max(worst["dw"], abs(float(w[0]) - float(dw_direct)))
                worst["spread"] = max(worst["spread"], abs(float(sp[0]) - float(d12)))
                l1 = 0.5 * (d1 + d2)
            for k, v in zip(keys, (loss[0], w[0], s[0], sp[0], d1, d2, l1)):
                acc[k] += float(v)
    n = len(batches)
    out = {k: v / n for k, v in acc.items()}
    out["identity_err_dw"] = worst["dw"]
    out["identity_err_spread"] = worst["spread"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="~/lisa-results")
    ap.add_argument("--tag", default=RUN_TAG)
    ap.add_argument("--seeds", default="1234,1")
    ap.add_argument("--sub-batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.replace(",", " ").split()]

    src = pathlib.Path(a.src).expanduser().resolve()
    manifest = json.loads((src / "manifest.json").read_text())
    ckpt_dir = src / "ckpt" / a.tag
    sweep = json.loads((src / "ov3" / f"val_wave_{a.tag}.json").read_text())
    hist = {arm: json.loads((src / f"history_{a.tag}_{arm}.json").read_text())[arm] for arm in ARMS}

    from huggingface_hub import snapshot_download
    local = snapshot_download("sanchit-gandhi/vctk", repo_type="dataset", allow_patterns=["data/*.parquet"])
    rows = build_index(local, cache=src / "train_index.json", verbose=False)
    keep, lens = corpus_from_index(rows, manifest)
    plan = draw_val(len(keep), lens)
    need = sorted({int(u) for idx, _ in plan for u in idx})
    cache = decode(local, keep, need, verbose=False)

    cwd = pathlib.Path.cwd(); os.chdir(src)
    try:
        G = boot(device="cpu", root=src)
    finally:
        os.chdir(cwd)
    import torch
    torch.set_num_threads(a.threads)
    from fast.convert_ckpt import build, import_module
    vb_det = make_vb(plan, cache, torch, device=torch.device("cpu"))
    vb = make_vb(plan, cache, torch, device=torch.device("cpu"), sub=a.sub_batch)

    res = {}
    print(f"\nseeds {seeds}; identity tolerance is float32 accumulation over 48000 samples\n", flush=True)
    for arm in sorted(ARMS):
        module, ck = G["load_arm"](ckpt_dir / f"{arm}.pt")
        stack, _, spec = build(G, arm, torch.device("cpu"))
        import_module(stack, module.cpu(), torch)
        batches = vb_det if stack.is_det else vb
        res[arm] = {"cls": spec[2], "is_det": bool(stack.is_det), "seeds": {}}
        for sd in seeds:
            t0 = time.time()
            r = decompose(G, stack, batches, sd, torch)
            anchor = sweep[arm]["runs"].get(str(sd))
            r["sweep_val_wave"] = anchor[1] if anchor else None
            r["anchor_err"] = abs(r["wave"] - anchor[1]) if anchor else None
            res[arm]["seeds"][sd] = r
            print(f"  {arm:<18} seed {sd:<5} val_wave {r['wave']:.9f}  spread {r['spread']:.9f}  "
                  f"L1_single {r['l1']:.9f}  | identity |dw| {r['identity_err_dw']:.1e} |sp| {r['identity_err_spread']:.1e}"
                  f"  | vs sweep {('%.1e' % r['anchor_err']) if anchor else '--':>8}  [{time.time()-t0:.0f}s]", flush=True)
        res[arm]["train_spread_final"] = hist[arm]["spread"][-1]
        res[arm]["recorded_val_wave"] = hist[arm]["val_wave"][-1]

    out = src / "ov3" / f"val_wave_decompose_{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))

    s0 = seeds[0]
    print(f"\n{'arm':<18} {'val_wave':>11} {'val spread':>11} {'sp/vw':>6} {'L1 single':>11} "
          f"{'L1 (train-spread proxy)':>24} {'proxy err':>9}")
    print("-" * 96)
    for arm in sorted(ARMS, key=lambda k: res[k]["seeds"][s0]["l1"]):
        r = res[arm]["seeds"][s0]
        proxy = res[arm]["recorded_val_wave"] + 0.5 * res[arm]["train_spread_final"]
        print(f"{arm:<18} {r['wave']:>11.6f} {r['spread']:>11.6f} {r['spread']/r['wave']:>6.2f} {r['l1']:>11.6f} "
              f"{proxy:>24.6f} {100*(proxy-r['l1'])/r['l1']:>+8.2f}%")
    print("\nranking by val_wave (the column as printed):   " + " < ".join(sorted(ARMS, key=lambda k: res[k]["seeds"][s0]["wave"])))
    print("ranking by single-draw L1 (waveform error):    " + " < ".join(sorted(ARMS, key=lambda k: res[k]["seeds"][s0]["l1"])))
    if len(seeds) > 1:
        print(f"\nsingle-draw L1 across seeds (eps noise on the ranking):")
        for arm in sorted(ARMS):
            v = [res[arm]["seeds"][sd]["l1"] for sd in seeds]
            print(f"  {arm:<18} " + "  ".join(f"{x:.6f}" for x in v) + f"   spread of seeds {100*(max(v)-min(v))/np.mean(v):.3f}%")
    worst_id = max(max(r["identity_err_dw"], r["identity_err_spread"]) for a_ in res.values() for r in a_["seeds"].values())
    worst_anchor = max((r["anchor_err"] or 0) for a_ in res.values() for r in a_["seeds"].values())
    print(f"\nworst identity error {worst_id:.2e}   worst disagreement with the sweep's val_wave {worst_anchor:.2e}")
    print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
