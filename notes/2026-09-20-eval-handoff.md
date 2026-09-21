# OV50 evaluation handoff

Written 2026-09-20, at the end of the 8-arm 50-epoch run, for whoever evaluates it next.
Training is DONE. The rental is terminated. Everything below is CPU work.

## What exists

Eight checkpoints, ~1.29 MB each, copied off the instance to the user's Mac at `~/lisa-results/`:

    det_paper.pt  det.pt  es_marg.pt  es_erb_l0.001.pt
    es_erb_l0.01.pt  es_erb_l0.1.pt  es_dec_l0.01.pt  es_dec_erb_l0.1.pt

alongside `<arm>.log`, `history_OV50_<arm>.json`, `calibration.json` and `manifest.json`.

Run constants are in `fast/run_contract.py`: 104,950 steps, batch 64, seg 48,000, R = 4, six
MultiStepLR milestones. The corpus was 39,639 utterances / 37.3224 h = 49.99 epochs, fingerprinted
in `manifest.json` as g3 corpus `db078cef…` / batches `0d24acf5…`. All eight arms shared ONE batch
stream (`BATCH_TAG = "OV50/batches"`, never per-arm), so the comparison is paired step for step.

## THE BLOCKER: the checkpoint format gap

`overnight3/e4_eval.py` → `load_ov3(tag)` → `load_arm(p)` (`overnight2/c1_model.py:278`) expects:

    ck["model"]     nn.Module state_dict
    ck["arm"]       (kind, lam, cls_name) TUPLE
    ck["cls"]       "LISAS" | "LISASD"
    ck["n_noise"]

`fast/train_arm.py:126` `save()` writes something else:

    ck["params"]    stack.P dict (stacked form, NOT a module state_dict)
    ck["arm"]       the arm NAME as a plain string, e.g. "es_erb_l0.1"
    (no "cls", no "n_noise")

Two of these fail **silently**. Do not skip this:

1. `load_arm` does `ck["arm"][0].startswith("det")`. On the string `"det_paper"` that is `"d"`,
   which is False — so **det_paper and det would be evaluated at tau = 1 (stochastic)** instead of
   tau = 0. No exception. Plausible, wrong numbers.
2. Missing `"cls"` defaults to `LISAS`. The two `es_dec_*` arms are **LISASD** (4 decoder-noise
   channels, 101-wide first layer instead of 97). Loading them as LISAS is wrong; it may or may not
   raise on a shape mismatch.

### The conversion is mechanical — reuse what already exists

`ArmStack.export(a, module)` (`overnight3/e2b_fast.py:399`) copies stacked params into a plain
module, and `e2b_fast.py:703` already writes exactly the blob `load_arm` wants:

    save_ckpt({"model": m.state_dict(), "step": …, "history": h, "arm": (kind, lam, cls_name),
               "n_noise": m.n_noise, "n_dec": getattr(m, "n_dec", 0), "cls": cls_name,
               "batch": batch, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")

So: rebuild the module per arm (kind/lam/cls from `ARMS` in `fast/run_contract.py`), `export`, save
in that shape. **Verify** by checking that the converted module reproduces the arm's final
`val_wave` from its own history — a converter that loads the wrong weights will still run.

## Metric conventions — read before reporting any number

**LSD is not in dB.** `lsd_db` (`build_notebook.py:650`) computes `log10(|X|^2)` with **no factor
of 10**, so its values are decades of power. True dB = 10 x the returned value. The name lies.
DO NOT "fix" this — the no-10 form is what every number already in this repo uses, and changing it
silently breaks comparability with all prior runs.

Three LSD variants are live in the tree:

| function | transform | scale |
|---|---|---|
| `lsd_db` (build_notebook:650) — used by e4_eval | log10(\|X\|^2) | 1x |
| `lsd_standard` (audit/lisa_paper_protocol.py:87) | log10(\|X\|^2) | 1x |
| `lsd_lisa` (audit/lisa_paper_protocol.py:81) — LISA's released utils.py | log10(\|X\|^4) | **2x** |

`lsd_lisa` squares an already-squared spectrogram, AND RMSes over time-within-bin rather than
frequency-within-frame, AND adds 1e-8 inside the sqrt, AND caps at 10.0. LISA's published LSD
numbers carry all four. Do not put them beside ours without saying so; the `det_paper` arm exists
precisely so the comparison can be internal, where the convention cancels.

**SNR is genuine dB** (`10*log10` of a power ratio). But note `max(sum(e**2), 1e-20)`: perfect
reconstruction returns a FINITE 220–240 dB, not infinity. Anything in that band is a passthrough or
identity bug, not a result. The meaningful reference is the predictability ceiling (~20–25 dB), not
the analytic one.

## Ideal values

| metric | ideal | note |
|---|---|---|
| SNR | ~230 dB = sentinel | real target is the predictability ceiling, ~20–25 dB |
| LSD | 0.000 | exact; eps never bites when Sh == Sy |
| PIT histogram | flat, 1/(M+1) = 0.0588 | M = 16 draws → 17 bins |
| PIT `pit_end` | 2/(M+1) = **0.1176** | above = under-dispersed, below = over-dispersed |
| ViSQOL NSIM | 1.000 | mapping-free — prefer it over MOS |
| ViSQOL MOS-LQO | ~4.7–4.75, NOT 5.0 | mapping saturates; measure the `ceiling` condition |

The prior OV3 run recorded `pit_end = 0.211`, i.e. **1.79x the ideal — under-dispersed**, the
ensemble too narrow. Whether the ERB term pulls it toward 0.1176 (and whether lambda = 0.1
overshoots into over-dispersion) is the question this run was built to answer.

PIT ranks are per time-frequency bin, so N is huge but heavily CORRELATED. Naive binomial error
bars will be far too tight. Do not call a small deviation significant on sample size alone.

**ViSQOL speech mode is nearly blind to this task.** It resamples to 16 kHz (8 kHz Nyquist) while
the band the model must invent is 6–24 kHz — it sees 11% of it. Audio mode at 48 kHz sees 100%.
Report speech mode for comparability with OV2 and the literature; lead with audio mode.
Also check `VISQOL_SP_MAPPING`: without `ai_edge_litert` it silently falls back from the lattice
mapping to polynomial, and the numbers stop matching OV2.

## Cross-arm comparability

`fast/compare.py` exists for this. The trainer's log prints each arm's OWN objective, and the eight
objectives are different functions — det_paper is lambda = 0, det adds 0.01 x a spectral term, and
the -erb arms fold lambda x (ERB term) inside, so a larger lambda inflates the number by
construction. Ranking that column ranks the arms by which loss they were handed.

- **`val_wave`** — same waveform term for every arm. This is the cross-arm column.
- **own objective** — comparable only within a shared `(kind, lambda)` pair, which here means
  exactly two pairs: {es_marg, es_dec_l0.01} and {es_erb_l0.1, es_dec_erb_l0.1}.

CRPS and PIT need multiple draws. `det_paper` and `det` are deterministic: CRPS collapses to
absolute error and PIT is degenerate. They are the reference line, not competitors, on those two.

## Not present in this repo

**FAD.** No Fréchet Audio Distance code anywhere (grep fad/frechet: zero hits). It needs an
embedding network — VGGish, PANNs and CLAP disagree with each other — and it is unstable below a
few hundred clips. EVAL12 is twelve. That is a study-design decision before it is a coding task.
