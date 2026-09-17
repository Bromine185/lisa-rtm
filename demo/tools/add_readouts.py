"""Add the ensemble readouts to the demo fixtures.

The evaluation's winning condition is not a single draw: it is `logmean16` — the per-bin mean of
log|STFT| across 16 draws, the phase of draw 0, resynthesised, with the baseband passed through
(overnight3/e1_model.py::logmag_ensemble_readout). That is the object the perceptual judges prefer,
so the demo has to be able to play it. This renders, per speaker and stochastic arm:

    <arm>_logmean16.wav   the log-magnitude ensemble readout, baseband passed through
    <arm>_mean16.wav      the plain waveform ensemble mean, for contrast (it is the SNR-optimal readout)

and merges them into the audio manifest under files.arms[<arm>].{logmean16, mean16}.

    venv/bin/python demo/tools/add_readouts.py --ckpt-dir lisa_rtm_cache/ckpt/final --out web/public/assets/audio
"""
import argparse, json, pathlib, sys, time
import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="lisa_rtm_cache/ckpt/final")
    ap.add_argument("--out", default="web/public/assets/audio")
    ap.add_argument("--M", type=int, default=16)
    a = ap.parse_args()
    # boot() chdirs into a scratch directory, so every path has to be absolute before it runs
    out = pathlib.Path(a.out)
    out = out if out.is_absolute() else (REPO / out).resolve()
    ckpt_dir = pathlib.Path(a.ckpt_dir)
    ckpt_dir = ckpt_dir if ckpt_dir.is_absolute() else (REPO / ckpt_dir).resolve()
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text())

    from audit.boot import boot
    from audit.vctk_fixtures import load as load_audio
    G = boot()
    CFG, reconstruct = G["CFG"], G["reconstruct"]
    readout = G["logmag_ensemble_readout"]

    ckpts = {p.stem.split("_step")[0]: p for p in sorted(ckpt_dir.glob("*.pt"))}
    models = {}
    for arm, p in ckpts.items():
        m, ck = G["load_arm"](p, CFG)
        if ck["arm"][0].startswith("det"):
            continue                      # a deterministic arm has one output; there is nothing to average
        models[arm] = m
    print(f"{len(models)} stochastic arms, M={a.M} draws each", flush=True)

    t0 = time.time()
    for spk in man["speakers"]:
        d = out / spk["id"]
        y = load_audio(d / "truth.wav", CFG.fs_hi)
        for arm, m in models.items():
            draws = np.stack([reconstruct(m, y, CFG, tau=1.0, seed=s) for s in range(a.M)])
            lm = readout(draws, y, CFG, passthrough=True)
            mn = draws.mean(0)
            sf.write(d / f"{arm}_logmean16.wav", np.clip(lm, -1, 1), CFG.fs_hi, subtype="PCM_16")
            sf.write(d / f"{arm}_mean16.wav", np.clip(mn, -1, 1), CFG.fs_hi, subtype="PCM_16")
            e = spk["files"]["arms"].setdefault(arm, {})
            e["logmean16"] = f"{spk['id']}/{arm}_logmean16.wav"
            e["mean16"] = f"{spk['id']}/{arm}_mean16.wav"
        print(f"  {spk['id']}  {len(models)} arms x {a.M} draws  [{time.time()-t0:.0f}s]", flush=True)

    man["readouts"] = {"M": a.M, "logmean16": "per-bin mean of log|STFT| over M draws, phase of draw 0, "
                                              "resynthesised, baseband passed through",
                       "mean16": "waveform mean of M draws"}
    man_path.write_text(json.dumps(man, indent=1))
    n = sum(1 for _ in out.rglob("*.wav"))
    print(f"manifest updated: {n} wav files, {sum(p.stat().st_size for p in out.rglob('*.wav')):,} bytes")


if __name__ == "__main__":
    main()
