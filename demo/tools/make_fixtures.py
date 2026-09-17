"""Audio fixtures and speaker metadata for the demo page.

For each demo speaker: pick one cached VCTK utterance (2.5-4.5 s, past the shared elicitation
paragraph), write it three ways, and optionally render every checkpoint in a directory through
the repo's own `reconstruct()`:

    <out>/<spk>/truth.wav        48 kHz   the utterance as the corpus loader gives it (mono, peak 0.95)
    <out>/<spk>/input.wav        12 kHz   decimate(): resample_poly(y, 1, 4)
    <out>/<spk>/naive.wav        48 kHz   the input resampled straight back up (nothing above 6 kHz)
    <out>/<spk>/<arm>.wav        48 kHz   one draw, tau = 1, seed = 0
    <out>/<spk>/<arm>_tau0.wav   48 kHz   tau = 0 (deterministic arms write <arm>.wav only)
    <out>/manifest.json          speaker id / gender / age / accent / region, utterance id, seconds,
                                 transcript when the archive gave one, and every file above

    venv/bin/python demo/tools/make_fixtures.py [--ckpt-dir DIR] [--out demo/assets/audio] [--no-text] [--skip 4]

VCTK's shared text (the Rainbow Passage) runs to about utterance 024, so with the default --skip 4
several speakers may read the same sentence; --skip 24 picks from each speaker's own newspaper
sentences instead (fetch more than 10 files per speaker first).

Audio comes from audit/vctk_fixtures.py (cached FLACs under lisa_rtm_cache/audit/<spk>/).  The
transcript is `txt/<spk>/<spk>_<nnn>.txt` in the remote archive: one central-directory listing over
HTTP ranges, six tiny member reads, cached beside the FLAC so later runs never touch the network.
Re-running is safe: base files are rewritten byte-identically, and arm renders already in the
manifest survive unless the same arm is rendered again.
"""
import argparse, json, pathlib, re, sys, time

import numpy as np
import scipy.signal as sps
import soundfile as sf

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from audit.vctk_fixtures import DATA, RangeFile, fetch, load          # noqa: E402

SPEAKERS = ("p236", "p237", "p238", "p360", "p361", "p374")
PER_SPEAKER = 10
FS_HI, R = 48000, 4
FS_LO = FS_HI // R
MIN_S, MAX_S = 2.5, 4.5
SKIP = 4                    # indices 0-3 are usually the shared elicitation paragraph
TAU_DRAW, SEED = 1.0, 0


# ---- speaker metadata -----------------------------------------------------------------------------
def parse_speaker_info(path):
    """`ID AGE GENDER ACCENTS REGION COMMENTS` -> {id: {gender, age, accent, region}}.

    Columns are whitespace-separated and REGION may hold several words ("New Jersey"), so
    everything after ACCENTS is the region.  Parenthesised comments are dropped."""
    out = {}
    for line in pathlib.Path(path).read_text().splitlines():
        line = re.sub(r"\(.*?\)", " ", line).strip()
        t = line.split()
        if len(t) < 4 or t[0] == "ID":
            continue
        out[t[0]] = {"gender": t[2], "age": int(t[1]) if t[1].isdigit() else None,
                     "accent": t[3], "region": " ".join(t[4:])}
    return out


# ---- utterance choice -----------------------------------------------------------------------------
def seconds_of(path):
    info = sf.info(str(path))
    return info.frames / info.samplerate


def pick_utterance(paths, skip=SKIP):
    """First cached utterance at sorted index >= skip that lasts MIN_S..MAX_S; otherwise the longest
    under MAX_S past skip; otherwise the shortest of them all."""
    paths = sorted(paths)
    durs = [seconds_of(p) for p in paths]
    tail = list(zip(paths, durs))[skip:] or list(zip(paths, durs))
    for p, d in tail:
        if MIN_S <= d <= MAX_S:
            return p, d, f"first 2.5-4.5 s past index {skip}"
    under = [(p, d) for p, d in tail if d < MAX_S]
    if under:
        p, d = max(under, key=lambda t: t[1])
        return p, d, "fallback: longest under 4.5 s"
    p, d = min(zip(paths, durs), key=lambda t: t[1])
    return p, d, "fallback: shortest available"


def utt_id(path):
    return pathlib.Path(path).stem.replace("_mic1", "")           # p236_005


# ---- transcripts ----------------------------------------------------------------------------------
def fetch_transcripts(utts, budget_s=120):
    """{spk: utt_id} -> {spk: text}.  Cached at DATA/<spk>/<utt>.txt; the archive is listed once."""
    texts, missing = {}, {}
    for spk, u in utts.items():
        p = DATA / spk / f"{u}.txt"
        if p.exists():
            texts[spk] = p.read_text().strip()
        else:
            missing[spk] = u
    if not missing:
        return texts
    import zipfile
    t0 = time.time()
    try:
        zf = zipfile.ZipFile(__import__("io").BufferedReader(RangeFile(), buffer_size=1 << 20))
        want = {f"txt/{spk}/{u}.txt": spk for spk, u in missing.items()}
        found = {}                                            # member name -> speaker
        for n in zf.namelist():
            for suffix, spk in want.items():
                if n == suffix or n.endswith("/" + suffix):
                    found[n] = spk
        print(f"archive listed in {time.time() - t0:.1f} s; {len(found)}/{len(want)} transcripts present")
        for n, spk in found.items():
            if time.time() - t0 > budget_s:
                print("transcript budget exhausted; stopping"); break
            txt = zf.read(n).decode("utf-8", "replace").strip()
            (DATA / spk / f"{missing[spk]}.txt").write_text(txt + "\n")
            texts[spk] = txt
    except Exception as e:                                    # network is optional here
        print(f"transcripts unavailable ({type(e).__name__}: {e}); continuing without")
    return texts


# ---- writing --------------------------------------------------------------------------------------
def write_wav(path, y, fs):
    """16-bit PCM.  Returns (bytes, clipped samples)."""
    y = np.asarray(y, np.float64)
    clipped = int(np.sum(np.abs(y) > 1.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(y, -1.0, 1.0), fs, subtype="PCM_16")
    return path.stat().st_size, clipped


def base_files(y, spk_dir):
    x_lo = sps.resample_poly(y, 1, R)                                  # the repo's decimate()
    naive = sps.resample_poly(x_lo, R, 1)[:len(y)]
    return {"truth": (y, FS_HI), "input": (x_lo, FS_LO), "naive": (naive, FS_HI)}


def arm_name(ckpt):
    return re.sub(r"_step\d+$", "", pathlib.Path(ckpt).stem)


# ---- main -----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=pathlib.Path, default=None, help="render every *.pt in here")
    ap.add_argument("--out", type=pathlib.Path, default=REPO / "demo" / "assets" / "audio")
    ap.add_argument("--no-text", action="store_true", help="skip the transcript fetch")
    ap.add_argument("--skip", type=int, default=SKIP, help="first sorted utterance index eligible")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else (pathlib.Path.cwd() / a.out).resolve()
    ckpt_dir = a.ckpt_dir.resolve() if a.ckpt_dir else None
    out.mkdir(parents=True, exist_ok=True)

    # audio (fetch is idempotent: cached files are reused) and speaker metadata
    paths = fetch(SPEAKERS, per_speaker=PER_SPEAKER)
    by_spk = {s: [p for p in paths if p.parent.name == s] for s in SPEAKERS}
    info_path = DATA / "speaker-info.txt"
    info = parse_speaker_info(info_path) if info_path.exists() else {}
    if not info:
        print(f"warning: {info_path} missing; speaker metadata will be blank")

    old = {}
    if (out / "manifest.json").exists():
        try:
            old = {s["id"]: s for s in json.loads((out / "manifest.json").read_text())["speakers"]}
        except Exception:
            old = {}

    chosen, ys, total = {}, {}, 0
    print(f"{'spk':<5} {'utterance':<9} {'sec':>5}  {'sex':<3} {'age':>3}  {'accent':<14} {'region':<18} rule")
    for spk in SPEAKERS:
        if not by_spk[spk]:
            sys.exit(f"no cached audio for {spk} under {DATA}")
        p, d, rule = pick_utterance(by_spk[spk], a.skip)
        y = load(p, FS_HI)
        y = y[: (len(y) // R) * R]                    # whole input cells: len(truth) == R * len(input)
        ys[spk] = y
        m = info.get(spk, {})
        chosen[spk] = {"id": spk, "gender": m.get("gender"), "age": m.get("age"),
                       "accent": m.get("accent"), "region": m.get("region"),
                       "utterance": utt_id(p), "seconds": round(len(y) / FS_HI, 4)}
        print(f"{spk:<5} {chosen[spk]['utterance']:<9} {len(y)/FS_HI:5.2f}  {m.get('gender',''):<3} "
              f"{str(m.get('age','')):>3}  {m.get('accent',''):<14} {m.get('region',''):<18} {rule}")

    texts = {} if a.no_text else fetch_transcripts({s: chosen[s]["utterance"] for s in SPEAKERS})
    for spk, t in texts.items():
        chosen[spk]["text"] = t

    # base files
    for spk in SPEAKERS:
        files = {}
        for name, (sig, fs) in base_files(ys[spk], out / spk).items():
            n, clipped = write_wav(out / spk / f"{name}.wav", sig, fs)
            total += n
            files[name] = f"{spk}/{name}.wav"
            if clipped:
                print(f"  {spk}/{name}.wav: {clipped} samples clipped to +-1")
        # arms rendered by an earlier run survive if their files still exist
        arms = {}
        for arm, ent in (old.get(spk, {}).get("files", {}).get("arms", {}) or {}).items():
            if all((out / ent[k]).exists() for k in ("draw", "tau0")):
                arms[arm] = ent
        files["arms"] = arms
        chosen[spk]["files"] = files

    # arm renders
    arm_meta = {}
    if ckpt_dir:
        ckpts = sorted(ckpt_dir.glob("*.pt"))
        if not ckpts:
            print(f"no *.pt in {ckpt_dir}")
        else:
            from audit.boot import boot                          # chdir's to a temp dir; paths above are absolute
            t0 = time.time()
            G = boot()
            print(f"booted {G['CFG'].name} on {G['DEVICE']} in {time.time() - t0:.1f} s")
            import torch
            for ck_path in ckpts:
                arm = arm_name(ck_path)
                m, ck = G["load_arm"](ck_path, G["CFG"])
                det = m.tau == 0.0
                stored = ck.get("arm", (None, None, None))
                arm_meta[arm] = {"file": ck_path.name, "step": ck.get("step"), "cls": ck.get("cls"),
                                 "objective": stored[0], "lambda": stored[1], "n_noise": ck.get("n_noise"),
                                 "n_dec": ck.get("n_dec"), "deterministic": bool(det), "seed": SEED,
                                 "params": int(sum(p.numel() for p in m.parameters()))}
                print(f"arm {arm}: {ck_path.name}  objective {stored}  cls {ck.get('cls')}  "
                      f"step {ck.get('step')}  {'deterministic' if det else 'stochastic'}")
                for spk in SPEAKERS:
                    y = ys[spk]
                    t1 = time.time()
                    if det:
                        y0 = G["reconstruct"](m, y, G["CFG"], tau=0.0, seed=SEED)
                        n, c = write_wav(out / spk / f"{arm}.wav", y0, FS_HI); total += n
                        ent = {"draw": f"{spk}/{arm}.wav", "tau0": f"{spk}/{arm}.wav"}
                        clipped = c
                    else:
                        y1 = G["reconstruct"](m, y, G["CFG"], tau=TAU_DRAW, seed=SEED)
                        y0 = G["reconstruct"](m, y, G["CFG"], tau=0.0, seed=SEED)
                        n1, c1 = write_wav(out / spk / f"{arm}.wav", y1, FS_HI)
                        n0, c0 = write_wav(out / spk / f"{arm}_tau0.wav", y0, FS_HI)
                        total += n1 + n0
                        ent = {"draw": f"{spk}/{arm}.wav", "tau0": f"{spk}/{arm}_tau0.wav"}
                        clipped = c1 + c0
                    hb = G["snr_db"](y, y0) if "snr_db" in G else float("nan")
                    print(f"  {spk} {len(y)/FS_HI:4.2f} s in {time.time() - t1:4.1f} s   "
                          f"tau0 SNR vs truth {hb:6.2f} dB" + (f"   {clipped} clipped" if clipped else ""))
                    chosen[spk]["files"]["arms"][arm] = ent
                del m

    manifest = {"fs_hi": FS_HI, "fs_lo": FS_LO, "upsample": R, "tau_draw": TAU_DRAW, "seed": SEED,
                "paths_relative_to": "this manifest's directory",
                "speakers": [chosen[s] for s in SPEAKERS]}
    old_arms = {}
    if (out / "manifest.json").exists():
        try:
            old_arms = json.loads((out / "manifest.json").read_text()).get("arms", {})
        except Exception:
            pass
    used = {a for s in manifest["speakers"] for a in s["files"]["arms"]}
    manifest["arms"] = {a: v for a, v in {**old_arms, **arm_meta}.items() if a in used}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    n_wav = sum(1 for _ in out.rglob("*.wav"))
    tot_wav = sum(p.stat().st_size for p in out.rglob("*.wav"))
    print(f"\nwrote {out}/manifest.json: {len(SPEAKERS)} speakers, arms {sorted(used) or 'none'}, "
          f"transcripts for {sorted(texts) or 'none'}")
    print(f"{n_wav} wav files, {tot_wav:,} bytes on disk ({total:,} written this run)")


if __name__ == "__main__":
    main()
