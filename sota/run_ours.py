"""Our OV50 arms on the same inputs as sota/run_models.py, in the same output format.

    venv/bin/python sota/run_ours.py --set core --arms es_dec_erb_l0.1 det --seeds 8

Input is the Chebyshev 12 kHz file sota/make_inputs.py wrote, fed through encode/decode exactly as
fast/paper_split_eval.py does for --resampler cheby. Samplers draw at tau = 1 with seeds 0..M-1; deterministic
arms run once. Output: <work>/<set>/out/ours_<arm>/<utt>_s<k>.npy.
"""
import argparse, json, pathlib, sys, time
import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
WORK = pathlib.Path("/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/work")

ap = argparse.ArgumentParser()
ap.add_argument("--set", default="core")
ap.add_argument("--arms", nargs="+", default=["es_dec_erb_l0.1", "det"])
ap.add_argument("--seeds", type=int, default=8)
ap.add_argument("--ckpt-dir", default="~/lisa-results/ckpt/OV50")
a = ap.parse_args()
ws = WORK / a.set
utts = json.load(open(ws / "utts.json"))["utts"]
from audit.boot import boot                                          # noqa: E402  (chdirs; paths above are absolute)
import torch                                                          # noqa: E402
G = boot(); CFG = G["CFG"]
# the ENV arms' class (LISASDW) lives in fast/arms_env.py; exec it so load_arm() can resolve the name
exec(compile((REPO / "fast/arms_env.py").read_text(), "<fast/arms_env.py>", "exec"), G)
for arm in a.arms:
    m = G["load_arm"](str(pathlib.Path(a.ckpt_dir).expanduser() / f"{arm}.pt"), CFG)[0]
    out = ws / "out" / f"ours_{arm}"; out.mkdir(parents=True, exist_ok=True)
    seeds = range(a.seeds) if m.tau != 0 else range(1)
    t0, audio = time.time(), 0.0
    for u in utts:
        x_lo, _ = sf.read(str(ws / "lo12k" / f"{u['utt']}.wav"), dtype="float64")
        n = int(round(u["seconds"] * 48000))
        for s in seeds:
            f = out / f"{u['utt']}_s{s}.npy"
            if f.exists():
                continue
            xt = torch.from_numpy(np.asarray(x_lo, np.float32))[None]
            with torch.no_grad():
                eps = None if m.tau == 0 else m.sample_eps(xt, 1.0, s)
                z = m.encode(xt, eps); n_out = xt.shape[1] * m.R
                w = torch.cat([m.decode(z, i, min(i + (1 << 15), n_out)) for i in range(0, n_out, 1 << 15)], 1)[0].numpy()
            np.save(f, np.pad(w, (0, max(0, n - len(w))))[:n].astype(np.float32)); audio += n / 48000
    wall = time.time() - t0
    json.dump({"model": f"ours_{arm}", "seeds": len(seeds), "stochastic": m.tau != 0, "wall_s": wall,
               "audio_s": audio, "rtf": wall / max(audio, 1e-9)}, open(out / "_run.json", "w"), indent=1)
    print(arm, f"{wall:.0f}s", flush=True)
