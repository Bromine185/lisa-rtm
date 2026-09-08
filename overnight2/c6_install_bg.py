# ============================================================ OV2-6a start the ViSQOL build in the background
# pesq/torchmetrics install synchronously (seconds); the google/visqol Bazel build runs as a detached
# subprocess writing /content/visqol_build.log so the kernel stays free for the ladder in the meantime.
import subprocess, sys, os, time, shlex
print("python", sys.version.split()[0], flush=True)
subprocess.run(f"{sys.executable} -m pip install -q pesq torchmetrics", shell=True)
try:
    import visqol  # noqa
    print("visqol already importable; no build needed", flush=True)
    VISQOL_BUILD = None
except ImportError:
    script = f"""set -x
curl -fsSL -o /usr/local/bin/bazel https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 && chmod +x /usr/local/bin/bazel
rm -rf /content/visqol && git clone --depth 1 https://github.com/google/visqol /content/visqol
cat /content/visqol/.bazelversion || true
cd /content/visqol && {sys.executable} -m pip install .
echo BUILD_EXIT=$?
"""
    open("/content/visqol_build.sh", "w").write(script)
    VISQOL_BUILD = subprocess.Popen(["bash", "/content/visqol_build.sh"], stdout=open("/content/visqol_build.log", "w"), stderr=subprocess.STDOUT)
    print("visqol build started in background, pid", VISQOL_BUILD.pid, "-> /content/visqol_build.log", flush=True)
