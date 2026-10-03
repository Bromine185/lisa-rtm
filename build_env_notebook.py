"""Generate sampler/lisa_rtm_train_env.ipynb -- the three ENV arms on one Colab A100, resumable.

    python3 build_env_notebook.py

The notebook is thin on purpose: every definition lives in the repo (fast/train_env.py, fast/arms_env.py,
fast/train_arm.py, fast/stage_corpus.py) and the notebook only mounts Drive, fetches the code, stages the
corpus and runs the trainer as a subprocess, so what runs on Colab is exactly what is committed.
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# ENV arms: is the residual muffling in the objective or in the context?

Three arms, one change each against `es_dec_erb_l0.1` (the OV50 sampler), on the **same 104,950-step batch
stream** as OV50, so every comparison is paired:

| arm | class | objective | what it tests |
|---|---|---|---|
| `es_env_l0.1` | LISASD | + `ES_env`: high-band log energy per 5 ms frame, three sub-bands | the objective |
| `es_ctx_erb_l0.1` | LISASDW | OV50's | 22 ms of context (dilated residual stack, +19.7 k params) |
| `es_ctx_env_l0.1` | LISASDW | + `ES_env` | both |

**Predictions (written 2026-09-27, before training, in `fast/arms_env.py`):** P5 the deficit's correlation with
the utterance's high-band share goes from −0.6 to above −0.2 for `es_ctx_env`, and the loud-frame deficit on
p236/p238 closes to better than −3 dB. P6 context moves it more than the objective. P7 coherence stays < 0.03.

**Budget.** One arm is ≈ 4–6 h on an A100 (OV50 measured ~1 s/step for eight stacked arms). Checkpoints go to
Drive every 500 steps and the trainer resumes from them, so a disconnect costs at most 500 steps: **re-run the
training cell** and it continues. Set `STOP_AFTER` to end a session early on purpose (the lr schedule is
untouched; only the loop stops).

**Runtime:** A100, High-RAM. The staged corpus is 32 GB on the local disk and is rebuilt per session (~15–20 min).
""")

code(r"""
# ---- 0. Drive, paths ----
from google.colab import drive
drive.mount('/content/drive')
import os, pathlib, subprocess, sys, json, time
DRIVE = pathlib.Path('/content/drive/MyDrive/lisa_rtm')
ROOT = DRIVE / 'env_run'            # checkpoints, histories, exported models (persists)
CORPUS = pathlib.Path('/content/corpus')   # staged corpus (local disk, rebuilt per session)
REPO = pathlib.Path('/content/lisa-rtm')
ROOT.mkdir(parents=True, exist_ok=True)
print('run dir', ROOT, '| GPU:'); print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout)
""")

md(r"""
## 1. The code

Two ways in, tried in order: a zip of the repo at `MyDrive/lisa_rtm/lisa-rtm.zip` (made with
`git archive`, or any zip of the working tree), or a clone of the private GitHub repo with a token stored
as the Colab secret `GITHUB_TOKEN` (key icon in the left bar). Set `BRANCH` to the branch that carries
`fast/train_env.py`.
""")

code(r"""
# ---- 1. code: Drive zip first, GitHub clone second ----
BRANCH = 'claude/sota-model-diagnosis-822734'
GITHUB_REPO = 'Bromine185/lisa-rtm'
if not (REPO / 'fast' / 'train_env.py').exists():
    z = DRIVE / 'lisa-rtm.zip'
    if z.exists():
        import zipfile, shutil
        tmp = pathlib.Path('/content/_unzip'); shutil.rmtree(tmp, ignore_errors=True)
        zipfile.ZipFile(z).extractall(tmp)
        src = next(p.parent for p in tmp.rglob('fast/train_env.py'))
        shutil.move(str(src), str(REPO)); print('code from', z)
    else:
        from google.colab import userdata
        tok = userdata.get('GITHUB_TOKEN')
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', BRANCH,
                        f'https://{tok}@github.com/{GITHUB_REPO}.git', str(REPO)], check=True)
        print('code from GitHub', BRANCH)
assert (REPO / 'fast' / 'train_env.py').exists(), 'no fast/train_env.py -- wrong zip or branch'
os.chdir(REPO)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'soundfile', 'huggingface_hub', 'pyarrow'], check=True)
import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
""")

md(r"""
## 2. The corpus, and the proof it is OV50's

`fast/stage_corpus.py` builds the 37.3 h training corpus deterministically (sorted utterances, `resample_poly`,
peak 0.95, float32 memmaps) and prints two digests: one of the corpus bytes and one of the first 1000 batch
draws. Both must equal OV50's, recorded in `overnight3/results_OV50/manifest.json`; if they do, these arms
draw the same segments in the same order as the eight OV50 arms and the comparison is paired by construction.
""")

code(r"""
# ---- 2. stage (skipped if already staged this session) ----
if not (CORPUS / 'manifest.json').exists():
    t0 = time.time()
    subprocess.run([sys.executable, 'fast/stage_corpus.py', '--out', str(CORPUS), '--workers', str(os.cpu_count())], check=True)
    print(f'staged in {(time.time()-t0)/60:.1f} min')
man = json.loads((CORPUS / 'manifest.json').read_text())
ref = json.loads((REPO / 'overnight3/results_OV50/manifest.json').read_text())
for k in ('corpus', 'batches'):
    same = man['g3'][k] == ref['g3'][k]
    print(f"g3 {k}: {man['g3'][k]}  {'== OV50' if same else '!= OV50 ' + ref['g3'][k]}")
assert man['g3']['batches'] == ref['g3']['batches'], 'batch stream differs from OV50: the comparison would not be paired'
if man['g3']['corpus'] != ref['g3']['corpus']:
    print('WARNING: corpus bytes differ from OV50 (the Hub dataset changed?); the batch INDICES still match')
print(f"{man['n']} utterances, {man['hours']:.2f} h")
""")

md(r"""
## 3. Train

Sequential, one arm at a time, each a subprocess of `fast/train_env.py` (resumable; checkpoints in `ROOT`).
Re-run this cell after any disconnect. Order: the two-change arm first, so the headline result exists even if
the budget runs out.
""")

code(r"""
# ---- 3. train (re-run to resume) ----
ARMS_TO_RUN = ['es_ctx_env_l0.1', 'es_ctx_erb_l0.1', 'es_env_l0.1']
STOP_AFTER = None          # e.g. 40000 to end this session at that step; the next run resumes from there
CKPT_EVERY = 500

def done(arm):
    h = ROOT / f'history_OV50_{arm}.json'
    return (ROOT / 'ckpt' / f'{arm}.pt').exists()

for arm in ARMS_TO_RUN:
    if done(arm):
        print(f'{arm}: finished and exported, skipping'); continue
    cmd = [sys.executable, '-u', 'fast/train_env.py', '--arm', arm, '--corpus', str(CORPUS), '--out', str(ROOT),
           '--ckpt-every', str(CKPT_EVERY)] + (['--stop-after', str(STOP_AFTER)] if STOP_AFTER else [])
    print('>>', ' '.join(cmd), flush=True)
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as p:
        for line in p.stdout:
            if 'Warning' in line or 'warn(' in line: continue
            print(line, end='', flush=True)
    if p.returncode != 0:
        raise SystemExit(f'{arm} exited {p.returncode}')
    if STOP_AFTER:
        print(f'{arm}: paused at STOP_AFTER; set STOP_AFTER = None and re-run to finish'); break
print('ALL DONE' if all(done(a) for a in ARMS_TO_RUN) else 'not all arms finished yet')
""")

md(r"""
## 4. Where things stand

Last step and validation loss per arm, against OV50's `es_dec_erb_l0.1` at the same steps (its history is in
the repo). The objectives differ by the `ES_env` term, so `val_wave` (the waveform part of the energy score)
is the comparable column, not `val_loss`.
""")

code(r"""
# ---- 4. status ----
import matplotlib.pyplot as plt
ref = json.loads((REPO / 'overnight3/results_OV50/history_OV50_es_dec_erb_l0.1.json').read_text())
ref = ref.get('es_dec_erb_l0.1', ref)
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(ref['val_step'], ref['val_wave'], 'k--', lw=1, label='OV50 es_dec_erb_l0.1 (control)')
for arm in ARMS_TO_RUN:
    f = ROOT / f'history_OV50_{arm}.json'
    ck = ROOT / f'{arm}.pt'
    if f.exists():
        h = json.loads(f.read_text())[arm]
    elif ck.exists():
        h = torch.load(ck, map_location='cpu', weights_only=False)['hist']
    else:
        print(f'{arm}: not started'); continue
    print(f"{arm:18s} step {h['val_step'][-1] if h['val_step'] else 0:6d}  val_wave {h['val_wave'][-1] if h['val_wave'] else float('nan'):.6f}"
          f"  {'exported' if (ROOT / 'ckpt' / f'{arm}.pt').exists() else 'in progress'}")
    ax.plot(h['val_step'], h['val_wave'], lw=1.5, label=arm)
ax.set_xlabel('step'); ax.set_ylabel('val_wave (energy score, waveform term)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
print('exported checkpoints:', sorted(p.name for p in (ROOT / 'ckpt').glob('*.pt')) if (ROOT / 'ckpt').exists() else 'none yet')
""")

md(r"""
## 5. After training

The exported checkpoints in `MyDrive/lisa_rtm/env_run/ckpt/` are `load_arm()` blobs. Score them on the Mac
with the SOTA harness, the same scorer every number in the paper came from:

```
venv/bin/python sota/run_ours.py --set wide    --ckpt-dir <env_run/ckpt> --arms es_ctx_env_l0.1 es_ctx_erb_l0.1 es_env_l0.1
venv/bin/python sota/run_ours.py --set core    --ckpt-dir <env_run/ckpt> --arms ... --seeds 8
venv/bin/python sota/run_ours.py --set ourtest --ckpt-dir <env_run/ckpt> --arms ... --seeds 16
lisa_rtm_cache/visqolenv/bin/python sota/score.py --set wide --models ours_es_ctx_env_l0.1 ...
```

P5 is read off `ourtest` (p236–p238, loud-frame deficit) and the per-utterance correlation over `wide` + `ourtest`.
""")

nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "A100"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = REPO / "sampler" / "lisa_rtm_train_env.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out, f"({len(CELLS)} cells)")
