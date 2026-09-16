"""Generate sampler/lisa_rtm_train.ipynb -- the seven-arm OV3 training run as one self-contained notebook.

The notebook is the experiment: every definition it needs is inlined from the repo's own sources at build
time, so nothing is exec'd from Drive at run time and the committed .ipynb is exactly what ran.  Open it in
Colab from GitHub, pick an A100, Run all.

    python3 build_train_notebook.py
"""
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parent
CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ---- sources inlined at build time -------------------------------------------------------------------
_NB_SRC = [("".join(c["source"])) for c in json.loads((REPO / "lisa_rtm.ipynb").read_text())["cells"]
           if c["cell_type"] == "code"]


def nb_cell(marker):
    """The one code cell of lisa_rtm.ipynb containing `marker`."""
    hits = [s for s in _NB_SRC if marker in s]
    assert len(hits) == 1, f"{marker!r} matched {len(hits)} cells"
    return hits[0].strip("\n")


def repo_file(rel):
    return (REPO / rel).read_text().strip("\n")


def inlined(rel, text):
    """Prefix a one-line provenance comment naming the repo file the source came from."""
    return f"# ---- inlined verbatim from {rel} ----\n{text}"


# ============================================================ title
md(r"""
# Training the OV3 sampler arms — energy score, seven arms, one batch stream

Seven paired arms of an 88k-parameter LISA are trained on **identical batches** from one initialisation, so
every difference between them is caused by the arm and nothing else. The question is the one the 8 September
run left open: the sampler wins the proper score and loses the perceptual judges. Is that a fact about
samplers, or about this sampler's distance, its noise pathway, and the statistic read out of it?

Three moves, one model class:

1. **The geometry of $d$.** The energy score $\mathrm{ES}_d(P,y) = \mathbb{E}\,d(Y,y) - \tfrac12 \mathbb{E}\,d(Y,Y')$
   is strictly proper for the law of $\phi(y)$ whenever $d(a,b) = \lVert\phi(a) - \phi(b)\rVert$. Nothing says
   $\phi$ has to be the per-bin log-magnitude the 8 September sampler used. The spectral weight is raised
   tenfold; one arm restricts the waveform term to the low band; two arms add an aggregate-proper term on log
   ERB-band energies, the space ViSQOL's neurogram lives in.
2. **Where the noise enters.** `LISAS` feeds eight Gaussian channels at the 12 kHz input, so every 48 kHz
   output sample inside one input interval is a deterministic function of the same three latents. `LISASD`
   adds four Gaussian channels per *output* sample at the decoder input: noise at the rate of the fine
   structure it has to generate.
3. **The readout.** LSD is squared error in log-power, so its minimiser is the conditional mean of the
   *log*-magnitude, which the waveform ensemble mean does not estimate. The evaluation cells compute it
   directly from the draws.

Training uses the **stacked trainer**: arms of one class share one forward and one backward (parameters carry
a leading arm axis), with bf16 autocast, a compiled decoder MLP, pinned batches, fused Adam, CUDA streams and
no per-step host syncs. The dashboard below reports throughput, **GPU memory** and **per-epoch** statistics live.
""")

# ============================================================ 0 setup
md(r"""
## 0. Setup

Packages, device, a persistent cache (Drive on Colab), and seeded random streams. `stream(label)` gives an
independent numpy generator per component, so re-running one cell does not disturb another's draws.
""")
code(inlined("lisa_rtm.ipynb §0", nb_cell("import importlib, subprocess, sys")))

# ============================================================ 1 config
md(r"""
## 1. Config

One frozen dataclass. `PRESET` picks `FULL` when CUDA is present and `SMOKE` otherwise. The launch cell
overrides batch size and segment length; everything else (model dimensions, learning-rate schedule, gradient
clip, evaluation STFT basis) comes from here.
""")
code(inlined("lisa_rtm.ipynb §1", nb_cell("@dataclasses.dataclass(frozen=True)")))

# ============================================================ 2 spectral
md(r"""
## 2. Spectral primitives

Hann STFT with exact inversion, log-magnitude, and the resynthesis helper the log-magnitude ensemble readout
needs. A metric is never computed on a modified spectrogram: we always resynthesise to a waveform and
re-analyse.
""")
code(inlined("lisa_rtm.ipynb §2", nb_cell("def _hann(n):")))

# ============================================================ 3 metrics
md(r"""
## 3. Metrics

SNR, LSD, per-third-octave band energy ratio (the direct over-smoothing measure), fair-ensemble CRPS, PIT
ranks and spread–skill.
""")
code(inlined("lisa_rtm.ipynb §4", nb_cell("def snr_db(y, y_hat):")))

# ============================================================ 4 data utilities
md(r"""
## 4. Data utilities

`decimate` is the anti-aliased downsample that makes the high band unrecoverable; `load_utterance` and the
range-request fetcher are kept for the DataShare fixtures the evaluation cells read.
""")
code(inlined("lisa_rtm.ipynb §6", nb_cell("VCTK_URL = ")))

# ============================================================ 5 corpus
md(r"""
## 5. Corpus — full VCTK 0.92 (mic1) from the Hub

97 training speakers, 39,639 utterances, 37.3 h at 48 kHz. Held out: p236/p237/p238 (ours) and the paper's own
split (speaker id ≥ 350). The first run downloads ~11 GB of parquet into the runtime and takes a few minutes;
it is cached for the session.
""")
code("# ---- packages the Hub loader needs (present on Colab) ----\n"
     'ensure("huggingface_hub")\nensure("pyarrow")\n\n'
     + inlined("overnight/cell1_corpus.py", repo_file("overnight/cell1_corpus.py")))

# ============================================================ 6 LISA
md(r"""
## 6. LISA

The paper's architecture: a four-layer conv encoder (kernels 7,3,3,1; channels 16,32,64,32) whose latents each
see 11 input samples, and a five-layer ReLU MLP of width 144 that reads a relative coordinate plus the three
nearest latents and emits one amplitude at any continuous time coordinate. 86,881 parameters.
""")
code(inlined("lisa_rtm.ipynb §7", nb_cell("class LISAEncoder(nn.Module):")))

# ============================================================ 7 training helpers
md(r"""
## 7. Training helpers

Batch sampling, chunked full-utterance reconstruction, the naive polyphase baseline, and the retrying
checkpoint writer (Colab's Drive mount intermittently loses a directory it wrote to seconds earlier).
""")
_TRAIN = nb_cell("def sample_batch(utts, cfg, rng):")
code(inlined("lisa_rtm.ipynb §8 (helpers only)", _TRAIN[:_TRAIN.index("CKPT_PATH = RUN")].strip("\n")))

# ============================================================ 8 the sampler
md(r"""
## 8. The sampler — `LISAS`, the distances, the energy score

Eight Gaussian channels are concatenated to the input waveform, so only the first convolution changes: +896
weights, 87,777 parameters. With the noise at zero the network is LISA exactly, which is why the deterministic
arms are the same class run at $\tau = 0$ and every arm shares one initialisation.

The two-draw unbiased estimator is
$\tfrac12[d(y,\hat y_1) + d(y,\hat y_2)] - \tfrac12 d(\hat y_1,\hat y_2)$. `HostCorpus` keeps the corpus in host
RAM (a 37 h corpus on the GPU is 32 GB) and moves aligned slices per step. The sequential trainer of this file
is stripped: the stacked trainer below replaces it.
""")
_C1 = repo_file("overnight2/c1_model.py")
code(inlined("overnight2/c1_model.py (train_ov2 / time_ov2 stripped)",
             _C1[:_C1.index("def train_ov2(")].rstrip("\n") + "\n\n\n" + _C1[_C1.index("def load_arm("):]))

# ============================================================ 9 OV3 additions
md(r"""
## 9. OV3 — decoder-side noise, the joint and ERB geometries, the readout

`LISASD` adds four Gaussian channels per output sample at the decoder input (+576 weights, 88,353 total);
`copy_shared` copies a LISAS into it with the noise columns zeroed, so both classes start identical at
$\varepsilon = 0$. New distances: `d_logmag_l2` (Euclidean on the whole log-magnitude spectrogram, strictly
proper for its joint law), and `d_erb` on 32 triangular bands equally spaced on the ERB-rate scale.
`logmag_ensemble_readout` is the LSD-optimal readout: the per-bin mean of $\log\lvert Y\rvert$ across the draws,
the phase of draw 0, resynthesised, with the given baseband passed through.
""")
code(inlined("overnight3/e1_model.py", repo_file("overnight3/e1_model.py")))

# ============================================================ 10 trainer helpers
md(r"""
## 10. Validation loss and curves

Each arm's **own** objective on a fixed held-out set (speakers p236–p237, utterances disjoint from the
evaluation set), with fixed noise seeds and no anchor jitter, so successive evaluations differ only through
the weights. `plot_curves` redraws train-versus-validation, the loss terms and the probe metrics at every
checkpoint.
""")
code(inlined("overnight3/e2_trainer.py", repo_file("overnight3/e2_trainer.py")))

# ============================================================ 11 fast trainer
md(r"""
## 11. The stacked trainer

Seven arms run sequentially cost ~84 ms per arm-step on an A100 for ~0.7 TFLOP of arithmetic: the step is
launch- and sync-bound, not compute-bound. Here the arms of one class share **one** forward and **one**
backward — parameters carry a leading arm axis, the encoder becomes one grouped convolution per layer and the
decoder one batched matmul per layer. Losses are independent per arm, so a backward on their sum gives each
arm its own gradient; one Adam over the stacked tensors is one Adam per arm; clipping is per arm.

Also: bf16 autocast (losses stay fp32), TF32, a compiled decoder MLP, pinned non-blocking batches, fused Adam,
two CUDA streams for the two classes, one gather on a replicate-padded latent, target STFT features computed
once per step and shared by every arm, and no `.item()` between callbacks.
""")
code(inlined("overnight3/e2b_fast.py", repo_file("overnight3/e2b_fast.py")))

# ============================================================ 12 dashboard
md(r"""
## 12. Live dashboard

Renders into a single updating output: progress and epoch, throughput (ms/step, steps/s, samples/s, and
seconds of audio per second of compute), **GPU memory** (allocated / reserved / peak / total, free, and
utilisation), a per-arm table (training EMA, validation loss and its terms, probe SNR and high-band deficit at
$\tau = 0$ and $\tau = 1$), and a **per-epoch** table that grows as epochs complete. Updates are throttled to
one every two seconds so the Colab output does not flood.
""")
code(r"""
# ---- live training dashboard (GPU memory, throughput, per-arm and per-epoch statistics) ----
import time as _time

try:
    from IPython.display import display, update_display, HTML
    _HAS_IPY = True
except Exception:
    _HAS_IPY = False

_MONO = "font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px;line-height:1.45"


def _hms(s):
    if s is None or s != s:
        return "--:--:--"
    s = int(max(0.0, s))
    return "%d:%02d:%02d" % (s // 3600, (s // 60) % 60, s % 60)


def _num(v, spec="%.4f", dash="&mdash;"):
    if v is None or (isinstance(v, float) and v != v):
        return dash
    try:
        return spec % v
    except Exception:
        return str(v)


def _bar(frac, n=34):
    frac = 0.0 if (frac is None or frac != frac) else min(max(float(frac), 0.0), 1.0)
    k = int(round(frac * n))
    return "&#9608;" * k + "&#9617;" * (n - k)


def _td(x, style=""):
    return "<td style='padding:1px 7px;text-align:right;" + style + "'>" + str(x) + "</td>"


def _th(x):
    return "<th style='padding:1px 7px;text-align:right;border-bottom:1px solid #8884'>" + str(x) + "</th>"


def make_dashboard(tag, arms, every=2.0):
    '''Returns (on_step, on_epoch) for train_ov3_fast.  Re-renders at most once every `every` seconds.'''
    spec = {k: (arm3(v) if "arm3" in globals() else (v[0], v[1], v[2] if len(v) > 2 else "LISAS")) for k, v in arms.items()}
    names = list(arms)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(DEVICE)
    did = "ov3-dash-" + str(tag)
    state = {"last": 0.0, "shown": False, "epochs": [], "info": None}

    def _html(info):
        g = info["gpu"]
        a = info["arms"]
        vals = [a[k].get("val_loss") for k in names if a[k].get("val_loss") is not None]
        best = min(vals) if vals else None
        h = ["<div style='" + _MONO + "'>"]
        h.append("<div style='font-weight:600'>" + str(tag) + " &middot; " + str(len(names)) + " arms &middot; batch "
                 + str(info["batch"]) + " &times; " + _num(info["seg_s"], "%.1f") + " s &middot; " + dev + "</div>")
        ep, ept = info.get("epoch"), info.get("epochs_total")
        h.append("<div>" + _bar(info["frac"]) + "  " + str(info["step"]) + "/" + str(info["steps"])
                 + "  " + _num(100 * info["frac"], "%.1f") + "%"
                 + ("  &middot; epoch " + _num(ep, "%.2f") + " / " + _num(ept, "%.2f") if ep is not None else "")
                 + "  &middot; elapsed " + _hms(info["t_elapsed"]) + "  &middot; ETA " + _hms(info.get("eta_s")) + "</div>")
        h.append("<div>throughput: " + _num(info["ms_per_step"], "%.0f") + " ms/step &middot; "
                 + _num(1000.0 / info["ms_per_step"] if info["ms_per_step"] else float("nan"), "%.2f") + " steps/s &middot; "
                 + _num(info["samples_per_s"], "%.1f") + " samples/s &middot; "
                 + _num(info["audio_s_per_s"], "%.1f") + " audio-s per compute-s &middot; lr "
                 + _num(info["lr"], "%.2e") + "</div>")
        used = (g["alloc_gb"] / g["total_gb"]) if g.get("total_gb") else None
        h.append("<div>GPU mem: " + _num(g["alloc_gb"], "%.1f") + " alloc / " + _num(g["reserved_gb"], "%.1f")
                 + " reserved / " + _num(g["peak_gb"], "%.1f") + " peak / " + _num(g["total_gb"], "%.1f")
                 + " GB total &middot; free " + _num(g["free_gb"], "%.1f") + " GB &middot; util "
                 + _num(g.get("util_pct"), "%.0f") + "%  " + _bar(used, 18) + "</div>")
        h.append("<table style='border-collapse:collapse;margin-top:6px;" + _MONO + "'><tr>"
                 + "".join(_th(c) for c in ("arm", "class", "kind", "lambda", "train EMA", "val loss", "val wave",
                                            "val spec", "SNR t0", "SNR t1", "def t0", "def t1")) + "</tr>")
        for k in names:
            kind, lam, cls = spec[k]
            r = a[k]
            hl = "background:#2e7d3222;font-weight:600" if (best is not None and r.get("val_loss") == best) else ""
            h.append("<tr>" + _td(k, "text-align:left") + _td(cls) + _td(kind) + _td(_num(lam, "%g"))
                     + _td(_num(r.get("loss_ema"))) + _td(_num(r.get("val_loss")), hl) + _td(_num(r.get("val_wave")))
                     + _td(_num(r.get("val_spec"), "%.3f")) + _td(_num(r.get("snr0"), "%.2f"))
                     + _td(_num(r.get("snr1"), "%.2f")) + _td(_num(r.get("def0"), "%+.2f"))
                     + _td(_num(r.get("def1"), "%+.2f")) + "</tr>")
        h.append("</table>")
        if state["epochs"]:
            h.append("<div style='margin-top:6px;font-weight:600'>per epoch</div>")
            h.append("<table style='border-collapse:collapse;" + _MONO + "'><tr>"
                     + "".join(_th(c) for c in ("epoch", "step", "wall")) + "".join(_th(k) for k in names) + "</tr>")
            for row in state["epochs"]:
                h.append("<tr>" + _td(_num(row["epoch"], "%.2f")) + _td(row["step"]) + _td(_hms(row["t"]))
                         + "".join(_td(_num(row["train"][k]) + " / " + _num(row["val"][k])) for k in names) + "</tr>")
            h.append("</table><div style='color:#8888'>cell: train EMA / val loss at the epoch boundary</div>")
        h.append("</div>")
        return "".join(h)

    def _text(info):
        g = info["gpu"]
        return ("[%s] step %d/%d (%.1f%%)  epoch %s  %.0f ms/step  ETA %s  gpu %.1f/%.1f GB (peak %.1f)"
                % (tag, info["step"], info["steps"], 100 * info["frac"],
                   _num(info.get("epoch"), "%.2f"), info["ms_per_step"], _hms(info.get("eta_s")),
                   g["alloc_gb"], g["total_gb"], g["peak_gb"]))

    def _render(info, force=False):
        now = _time.time()
        if not force and now - state["last"] < every:
            return
        state["last"] = now
        if not _HAS_IPY:
            print(_text(info), flush=True)
            return
        if state["shown"]:
            update_display(HTML(_html(info)), display_id=did)
        else:
            display(HTML(_html(info)), display_id=did)
            state["shown"] = True

    def on_step(info):
        state["info"] = info
        _render(info, force=info["step"] >= info["steps"])

    def on_epoch(info):
        a = info["arms"]
        state["epochs"].append({"epoch": info.get("epoch"), "step": info["step"], "t": info["t_elapsed"],
                                "train": {k: a[k].get("loss_ema") for k in names},
                                "val": {k: a[k].get("val_loss") for k in names}})
        _render(info, force=True)

    return on_step, on_epoch


print("dashboard ready (make_dashboard); IPython display:", _HAS_IPY)
""")

# ============================================================ 13 launch
md(r"""
## 13. Launch

Seven arms, one batch stream, one initialisation per class. `det` and `es_marg` are the 8 September references
at $\lambda = 10^{-2}$; the five new arms sit at $\lambda = 10^{-1}$ and vary one thing each: the spectral
weight alone, the waveform term restricted to the low band, the ERB aggregate term, decoder-side noise, and
both together.

Set `BATCH_OVERRIDE`, `BUDGET_H` or `OV3_TAG` in a cell above to change the recipe. Memory scales with
batch × arms-per-stack, so `GROUP_MAX = 2` chunks each stack to two arms.
""")
code(r"""
# ---- the seven paired arms (overnight3/e3_launch.py) ----
ARMS = {
    "det":             ("det",               1e-2, "LISAS"),    # reference (8 Sep recipe)
    "es_marg":         ("es_marg",           1e-2, "LISAS"),    # reference sampler (8 Sep recipe)
    "es_marg_l0.1":    ("es_marg",           1e-1, "LISAS"),    # spectral term dominant (weight x10)
    "es_split_l0.1":   ("es_split_marg",     1e-1, "LISAS"),    # + waveform term on the low band only
    "es_erb_l0.1":     ("es_marg_erb",       1e-1, "LISAS"),    # + aggregate-proper term on log ERB-band energies
    "es_dec_l0.1":     ("es_marg",           1e-1, "LISASD"),   # noise at the output rate
    "es_dec_erb_l0.1": ("es_marg_erb",       1e-1, "LISASD"),   # both
}
if isinstance(globals().get("ARMS_OVERRIDE"), dict):
    ARMS = dict(ARMS_OVERRIDE)
    print("ARMS_OVERRIDE in effect:", list(ARMS), flush=True)

BATCH = int(globals().get("BATCH_OVERRIDE", 64))
SEG = int(globals().get("SEG_OVERRIDE", 48000))
OV3_TAG = globals().get("OV3_TAG", "OV3_fast")
BUDGET_H = float(globals().get("BUDGET_H", 3.0))
FAST["GROUP_MAX"] = globals().get("GROUP_MAX", 2)

import gc, sys, json
for _n in ("models_ov3", "hist_ov3"):
    if _n in globals():
        del globals()[_n]
sys.last_traceback = sys.last_value = sys.last_type = None
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print("GPU before corpus: %.2f GB allocated, %.2f GB reserved"
          % (torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9), flush=True)

if "corpus" not in globals():                      # reuse the host-RAM corpus if a previous cell built it
    corpus = HostCorpus(train_utts, CFG, seg_hi=SEG)
    val_corpus = HostCorpus(test_utts[12:52], CFG, seg_hi=SEG)   # test speakers, utterances disjoint from EVAL12
    _step = max(1, len(train_utts) // 200)          # the 200-utterance fit subset; the full list is 26 GB of RAM
    train_utts, train_spk = train_utts[::_step][:200], train_spk[::_step][:200]
    gc.collect()
else:
    print("reusing corpus / val_corpus from the kernel", flush=True)

steps_per_epoch = corpus.hours * 3600 / (BATCH * SEG / CFG.fs_hi)
dt = time_ov3_fast(corpus, ARMS, BATCH)
steps = max(2000, (int(BUDGET_H * 3600 / dt) // 1000) * 1000)
if isinstance(globals().get("STEPS_OVERRIDE"), int):
    steps = int(STEPS_OVERRIDE)
    print("STEPS_OVERRIDE in effect:", steps, flush=True)
plan = ("plan: %d arms x %d steps = %.1f epochs of %.1f h; est %.2f h at %.0f ms/step; %.0f steps/epoch; "
        "batch %d x %.1f s; val %d utts"
        % (len(ARMS), steps, steps / steps_per_epoch, corpus.hours, steps * dt / 3600, dt * 1000,
           steps_per_epoch, BATCH, SEG / CFG.fs_hi, val_corpus.n))
print(plan, flush=True)
(ROOT / ("train_%s.log" % OV3_TAG)).write_text(plan + chr(10) + json.dumps(ARMS) + chr(10))

on_step, on_epoch = make_dashboard(OV3_TAG, ARMS)
models_ov3, hist_ov3 = train_ov3_fast(corpus, val_corpus, ARMS, steps, BATCH, 1e-3,
                                      (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3, 1000, OV3_TAG, test_utts[0],
                                      on_step=on_step, on_epoch=on_epoch, steps_per_epoch=steps_per_epoch)
json.dump({k: hist_ov3[k] for k in ARMS}, open(ROOT / ("ov3_history_%s.json" % OV3_TAG), "w"))
print("TRAINING DONE", OV3_TAG, flush=True)
""")

# ============================================================ 14 evaluation
md(r"""
## 14. Evaluation — CRPS, calibration, coherence, and the readouts

Every arm on the held-out Hub set: one draw and the 16-draw ensemble (CRPS, sliced CRPS, PIT, spread–skill,
ensemble-mean metrics, coherent fraction and $\kappa$), the log-magnitude ensemble readouts, and every
condition again with the given baseband passed through.
""")
code("OV3_RUN_EVAL = True\nOV3_TAG = globals().get(\"OV3_TAG\", \"OV3_fast\")\n\n"
     + inlined("overnight3/e4_eval.py", repo_file("overnight3/e4_eval.py")))

md(r"""
## 15. Perceptual evaluation — LSD and ViSQOL

The user's `evaluation.py` call path: ViSQOL speech mode at 16 kHz, ViSQOL audio mode at 48 kHz (which sees the
whole reconstructed band), and wideband PESQ, for every condition with and without passthrough, plus the floor
(passthrough with an empty high band) and ceiling (passthrough with the true high band) rows. Requires the
ViSQOL install cells to have been run in this kernel.
""")
code("OV3_RUN_EVAL = True\nOV3_TAG = globals().get(\"OV3_TAG\", \"OV3_fast\")\n\n"
     + inlined("overnight3/e5_visqol.py", repo_file("overnight3/e5_visqol.py")))


nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = REPO / "sampler" / "lisa_rtm_train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(CELLS)} cells, {sum(c['cell_type'] == 'code' for c in CELLS)} code)")
