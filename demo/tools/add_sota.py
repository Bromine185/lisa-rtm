"""Fold the released models' outputs on the demo utterances into the demo's audio assets.

    venv/bin/python demo/tools/add_sota.py [--work lisa_rtm_cache/sota/work/demo] [--out web/public/assets/audio] [--manifest-only]

Reads <work>/out/<model>/<utt>_s<k>.npy (sota/run_models.py on the `demo` set: the demo's six utterances,
the demo's own resample_poly input, no trimming), and writes

    <out>/<spk>/<model>.wav          48 kHz, 16-bit, draw 0 (the only output of a deterministic model)
    <out>/<spk>/<model>_draw2.wav    48 kHz, draw 1, samplers only

Each file is gain-aligned to the truth on the band every model was handed (energy below 5.5 kHz), exactly
as sota/score.py aligns before scoring, and the applied gain is recorded: FLowHigh peak-normalises to 0.99
and AudioSR normalises its input, so without this the A/B would compare levels, not bands.

The manifest gains a top-level `released` block (per model: name, family, params, det, draws, seen
speakers and the basis of that list, steps, RTF, device and machine, and `arch`: the architecture facts
with per-block parameter counts) and, per speaker, `files.released[model] = {draw, draw2?, gain}`.
Model facts come from demo/tools/sota_models.json (no measured numbers), parameter counts from
sota/params.json (sota/count_params.py), RTFs from the scored core run's _run.json (sota/run_models.py).
Re-running overwrites the same files and entries and touches nothing else. --manifest-only rewrites the
`released` block alone: no audio is read or written.
"""
import argparse, json, pathlib, platform, subprocess, sys

import numpy as np
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
FS, ALIGN = 48000, 5500.0
MODELS = ["flowhigh", "apbwe", "nuwave2", "audiosr"]
CORE = pathlib.Path("/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/work/core")


def this_machine():
    # _run.json files written before run_models.py recorded the machine: the cache is this Mac's, so name it
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or None


def run_meta(m):
    """RTF, device and machine. First choice: sota/rtf.json (sota/bench_rtf.py: the model alone on an idle
    machine, warm, median of repeats). Fallback: the scored core run's _run.json (sota/run_models.py), whose
    wall clock includes file I/O and may have shared the machine; a _run.json with no audio in it (a resumed
    run that found every draw on disk) carries no timing: None."""
    b = pathlib.Path(__file__).resolve().parents[2] / "sota" / "rtf.json"
    if b.exists():
        r = json.loads(b.read_text()).get(m)
        if r and r.get("rtf"):
            return round(float(r["rtf"]), 4), r.get("device"), r.get("machine") or this_machine()
    q = CORE / "out" / m / "_run.json"
    if not q.exists():
        return None, None, None
    r = json.loads(q.read_text())
    ok = (r.get("audio_s") or 0) > 0 and (r.get("draws_run") or 0) > 0
    return (round(float(r["rtf"]), 4) if ok else None), r.get("device"), r.get("machine") or this_machine()


def arch_of(f, c):
    """The facts as the page shows them, with each block's parameter count from sota/params.json."""
    a = {k: v for k, v in f.items() if k not in ("blocks", "count_whole")}
    a["params"] = c["total"]
    a["blocks"] = [{"id": b["id"], "label": b["label"], "detail": b["detail"], "params": c["blocks"][b["id"]]} for b in f["blocks"]]
    return a


def align_gain(y, w):
    n = min(len(y), len(w)); Y = np.fft.rfft(y[:n]); W = np.fft.rfft(w[:n]); lo = np.fft.rfftfreq(n, 1 / FS) < ALIGN
    return float(np.sqrt(np.sum(np.abs(Y[lo]) ** 2) / max(np.sum(np.abs(W[lo]) ** 2), 1e-30)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(pathlib.Path("/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota/work/demo")))
    ap.add_argument("--out", default=str(REPO / "web/public/assets/audio"))
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--manifest-only", action="store_true", help="rewrite manifest.released only; no audio read or written")
    a = ap.parse_args()
    work, out = pathlib.Path(a.work), pathlib.Path(a.out)
    man_p = out / "manifest.json"
    man = json.loads(man_p.read_text())
    facts = json.loads((REPO / "demo/tools/sota_models.json").read_text())["models"]
    counts = json.loads((REPO / "sota/params.json").read_text())

    released = {}
    for m in a.models:
        f, c = facts[m], counts[m]
        rtf, dev, mach = run_meta(m)
        released[m] = {"name": f["name"], "family": f["family"], "params": c["total"], "det": f["det"],
                       "draws": 1 if f["det"] else 2, "seen": f["seen"], "seen_basis": f.get("seen_basis", "stated"),
                       "seen_note": f["seen_note"], "steps": f["steps"], "rtf_m4": rtf, "rtf_device": dev, "machine": mach,
                       "arch": arch_of(f, c)}
    if a.manifest_only:
        old = man.get("released") or {}
        man["released"] = {**old, "models": {**(old.get("models") or {}), **released}}
        man_p.write_text(json.dumps(man, indent=1))
        print("manifest.released rewritten (no audio touched):", man_p, flush=True)
        return
    utts = {u["utt"]: u for u in json.loads((work / "utts.json").read_text())["utts"]}
    by_spk = {u.split("_")[0]: u for u in utts}
    for sp in man["speakers"]:
        utt = by_spk.get(sp["id"])
        if not utt:
            continue
        y = np.load(work / "truth" / f"{utt}.npy")
        sp["files"].setdefault("released", {})
        for m in a.models:
            files = sorted((work / "out" / m).glob(f"{utt}_s*.npy"), key=lambda q: int(q.stem.rsplit("_s", 1)[1]))
            if not files:
                print(f"  {sp['id']} {m}: no output", flush=True); continue
            entry = {}
            for k, q in enumerate(files[: released[m]["draws"]]):
                w = np.load(q).astype(np.float64)[: len(y)]
                g = align_gain(y, w); w = np.clip(w * g, -1, 1)
                name = f"{m}.wav" if k == 0 else f"{m}_draw{k + 1}.wav"
                sf.write(str(out / sp["id"] / name), w.astype(np.float32), FS, subtype="PCM_16")
                entry["draw" if k == 0 else f"draw{k + 1}"] = f"{sp['id']}/{name}"
                entry.setdefault("gain", round(g, 4))
            sp["files"]["released"][m] = entry
            print(f"  {sp['id']} {m}: {len(entry) - 1} file(s), gain {entry['gain']:.3f}", flush=True)
    man["released"] = {"models": released, "input": "the demo's own input (resample_poly 48 kHz -> 12 kHz), the same file the OV50 arms get",
                       "gain": f"each output scaled so its energy below {ALIGN:.0f} Hz matches the truth's, as sota/score.py does before scoring",
                       "source": "sota/run_models.py on the `demo` set; demo/tools/add_sota.py"}
    man_p.write_text(json.dumps(man, indent=1))
    print("manifest updated:", man_p, flush=True)


if __name__ == "__main__":
    sys.exit(main())
