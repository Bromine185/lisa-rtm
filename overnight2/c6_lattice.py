# ============================================================ OV2-6c try the upstream-default speech MOS mapping
import subprocess, sys, importlib
r = subprocess.run(f"{sys.executable} -m pip install -q 'visqol-python[lattice]'", shell=True, text=True, capture_output=True)
print((r.stdout or "")[-600:], (r.stderr or "")[-900:], "[pip rc", r.returncode, "]", flush=True)
try:
    import ai_edge_litert; print("ai-edge-litert available: speech MOS will use the lattice model (upstream default)", flush=True)
except Exception as e:
    print("lattice runtime NOT available on this Python; speech MOS uses the polynomial mapping; NSIM is recorded as the mapping-free score:", repr(e), flush=True)
