"""Execute the notebook's code cells end-to-end without touching the network.

The VCTK cell is replaced with synthetic speech-like audio (harmonic stack + fricative noise) so
every other cell -- transport, model, training, gates, controls, stochastic rung, figures -- runs
for real.  This is a smoke test of the code path, not a scientific result.

    ./venv/bin/python validate.py
"""
import json
import os
import sys
import types
import pathlib

os.environ.setdefault("MPLBACKEND", "Agg")

NB = json.loads((pathlib.Path(__file__).parent / "lisa_rtm.ipynb").read_text())
CODE = [c for c in NB["cells"] if c["cell_type"] == "code"]
SRC = ["".join(c["source"]) for c in CODE]

NAMES = ["setup", "config", "spectral", "transport", "metrics", "tests(gate1)",
         "data_utils", "data_build",
         "model", "training", "gate2", "ladder", "controls", "stochastic", "latency",
         "listen", "results"]
assert len(SRC) == len(NAMES), f"expected {len(NAMES)} code cells, notebook has {len(SRC)}"

# Outside a notebook, display() has nothing to render into; neutralise it.
import IPython.display as _disp
_disp.display = lambda *a, **k: None

G = {"__name__": "__main__"}


def synthetic_corpus(cell_src):
    """Replace the VCTK fetch with speech-like signals at CFG.fs_hi."""
    import numpy as np
    cfg = G["CFG"]
    stream = G["stream"]

    def utterance(label, seconds=1.6):
        rng = stream(f"synth/{label}")
        fs = cfg.fs_hi
        n = int(seconds * fs)
        t = np.arange(n) / fs
        f0 = 110 + 40 * rng.standard_normal() + 15 * np.sin(2 * np.pi * 1.7 * t)
        phase = 2 * np.pi * np.cumsum(f0) / fs
        # harmonic stack with a formant-ish rolloff -- predictable low band, rich high band
        x = sum((1.0 / h ** 0.9) * np.sin(h * phase + rng.uniform(0, 2 * np.pi))
                for h in range(1, 60))
        # fricative bursts: broadband, phase-unpredictable, exactly what regresses to the mean
        env = np.zeros(n)
        for _ in range(4):
            s = rng.integers(0, n - fs // 8)
            env[s:s + fs // 8] = np.hanning(fs // 8)
        x = x * (0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t)) + 3.0 * env * rng.standard_normal(n)
        # silence at the edges, so the silence-inflation failure mode is reachable
        x[:fs // 10] = 0.0
        x[-fs // 10:] = 0.0
        m = np.max(np.abs(x))
        return (x / m * 0.95).astype(np.float32)

    G["train_utts"] = [utterance(f"train/{i}") for i in range(8)]
    G["train_spk"] = ["synth"] * 8
    G["test_utts"] = [utterance(f"test/{i}") for i in range(4)]
    G["test_spk"] = ["synth"] * 4
    print(f"SYNTHETIC corpus: {len(G['train_utts'])} train / {len(G['test_utts'])} test "
          f"utterances at {cfg.fs_hi} Hz")


failures = []
for i, (name, src) in enumerate(zip(NAMES, SRC)):
    print(f"\n{'='*70}\n[{i}] {name}\n{'='*70}")
    try:
        if name == "data_build":
            synthetic_corpus(src)
        else:
            exec(compile(src, f"<cell {i} {name}>", "exec"), G)
    except Exception:
        import traceback
        traceback.print_exc()
        failures.append(name)
        break

print("\n" + "=" * 70)
if failures:
    print(f"FAILED at cell: {failures}")
    sys.exit(1)
print("ALL CELLS EXECUTED")
