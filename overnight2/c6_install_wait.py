# ============================================================ OV2-6b ViSQOL via the pure-Python port + PESQ, self-test
# evaluation.py's `from visqol import VisqolApi` is the third-party pure-Python port `visqol-python`
# (PyPI, Python >= 3.10), not Google's Bazel package.  Stop any Bazel build started earlier and pip install.
import subprocess, sys, time, importlib, numpy as np, torch
if globals().get("VISQOL_BUILD") is not None:
    try:
        VISQOL_BUILD.kill(); print("killed the unnecessary Bazel build", flush=True)
    except Exception as e:
        print("bazel build already gone:", e)
    VISQOL_BUILD = None
subprocess.run("pkill -f bazel >/dev/null 2>&1; pkill -f visqol_build.sh >/dev/null 2>&1", shell=True)
r = subprocess.run(f"{sys.executable} -m pip install -q 'visqol-python[accel]' pesq torchmetrics", shell=True, text=True, capture_output=True)
print((r.stdout or "")[-800:], (r.stderr or "")[-800:], "[pip rc", r.returncode, "]", flush=True)
if r.returncode != 0:
    r = subprocess.run(f"{sys.executable} -m pip install -q visqol-python pesq torchmetrics", shell=True, text=True, capture_output=True)
    print("plain install:", (r.stdout or "")[-800:], (r.stderr or "")[-800:], "[pip rc", r.returncode, "]", flush=True)
importlib.invalidate_caches()
import visqol
from visqol import VisqolApi
print("visqol port version:", getattr(visqol, "__version__", "?"), flush=True)
_api = VisqolApi(); _api.create(mode="speech")
_ref = np.random.default_rng(0).standard_normal(16000 * 3) * 0.1
print("visqol self-test speech16k ref-vs-ref:", _api.measure_from_arrays(_ref, _ref, sample_rate=16000).moslqo, flush=True)
try:
    _api2 = VisqolApi(); _api2.create(mode="audio")
    _ref48 = np.random.default_rng(0).standard_normal(48000 * 3) * 0.1
    print("visqol self-test audio48k ref-vs-ref:", _api2.measure_from_arrays(_ref48, _ref48, sample_rate=48000).moslqo, flush=True)
except Exception as e:
    print("audio mode unavailable in this port:", repr(e), flush=True)
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
_p = PerceptualEvaluationSpeechQuality(16000, "wb"); _t = torch.randn(16000 * 3) * 0.1
print("pesq self-test ref-vs-ref:", float(_p(_t, _t)), flush=True)
print("INSTALL DONE", flush=True)
