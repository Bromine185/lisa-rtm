# Scale freedom, all seven arms — `OV3_fast`, step 16000

**Date:** 2026-09-17. **Data:** 6 held-out utterances — `p236_002 p237_001 p238_001 p360_001 p361_001 p374_001`, mic1. **Draw:** τ = 0 and τ = 1, seed 0 (`det` is τ = 0 only). **Measurement:** encode once at 12 kHz, decode the *same* latents at ×4 and at ×8; decimate the ×8 output by 2 with `resample_poly` and take the SNR against the direct ×4 output — that is the agreement. "Above 24 kHz" is the share of the ×8 query's energy at or over the ×4 Nyquist. The curvature ratio is mean |d²f/dc²| in the gaps between the four trained coordinate phases divided by mean |d²f/dc²| at them, over an 801-point sweep of c ∈ [−1, 1] at the loudest latent triple; a decoder that memorised its four phases is ≫ 1. The two LISASD arms draw decoder noise per output sample, so the ×8 lattice needs a rule: `held` repeats each ×4 noise row for two ×8 samples, `fresh` redraws at ×8. `held` is the headline; both are measured.

**Reproduce:** `venv/bin/python audit/scale_freedom_all.py` → `lisa_rtm_cache/results/scale_OV3_fast.json` + `_sweeps.npz`; `venv/bin/python audit/scale_freedom_all_figs.py` → the three figures below. Extends [`../2026-09-17-snr-ceiling-gating-and-scale.md`](../2026-09-17-snr-ceiling-gating-and-scale.md) §3, which had one arm, one utterance, step 13500.

## 1. The table

| arm | class | agreement dB, τ=0 | agreement dB, τ=1 | above 24 kHz %, τ=0 | above 24 kHz %, τ=1 | curvature ratio, τ=0 | curvature ratio, τ=1 |
|---|---|---:|---:|---:|---:|---:|---:|
| det | LISAS | 44.21 / 40.84 | — | 0.0048 / 0.0089 | — | 2.32 / 5.24 | — |
| es_marg | LISAS | 40.34 / 33.37 | 38.75 / 33.12 | 0.0173 / 0.0545 | 0.0212 / 0.0573 | 1.32 / 2.20 | 1.34 / 2.89 |
| es_marg_l0.1 | LISAS | 45.58 / 39.70 | 44.14 / 38.60 | 0.0041 / 0.0114 | 0.0055 / 0.0148 | 2.15 / 5.05 | 2.13 / 4.17 |
| es_split_l0.1 | LISAS | 46.54 / 40.34 | 46.44 / 40.51 | 0.0035 / 0.0099 | 0.0034 / 0.0095 | 1.97 / 5.35 | 1.44 / 4.16 |
| es_erb_l0.1 | LISAS | 46.11 / 38.02 | 39.41 / 32.93 | 0.0046 / 0.0165 | 0.0184 / 0.0552 | 2.05 / 7.17 | 1.29 / 2.17 |
| es_dec_l0.1 | LISASD | 46.48 / 39.19 | 42.37 / 37.47 | 0.0039 / 0.0129 | 0.0077 / 0.0192 | 1.77 / 5.64 | 2.15 / 7.73 |
| es_dec_erb_l0.1 | LISASD | 47.27 / 40.96 | 38.38 / 34.88 | 0.0029 / 0.0086 | 0.0186 / 0.0350 | 1.71 / 2.57 | 2.48 / 6.05 |

Each cell is **mean / worst** over the six utterances. Worst is the minimum for agreement and the maximum for the other two. The curvature mean is dragged up by a few cells where the denominator nearly vanishes; the medians are det 1.28, es_marg 1.15 / 1.06, es_marg_l0.1 2.00 / 1.69, es_split_l0.1 1.32 / 0.93, es_erb_l0.1 1.09 / 0.97, es_dec_l0.1 1.02 / 1.21, es_dec_erb_l0.1 1.77 / 1.55 (τ=0 / τ=1). Pooled over all 78 arm × τ × utterance cells the median is 1.18 and 41 % sit below 1.

### 1.1 Decoder noise on the ×8 lattice, τ = 1

| arm | agreement, held | agreement, fresh | above 24 kHz %, held | fresh | even-sample match, held | fresh |
|---|---:|---:|---:|---:|---:|---:|
| es_dec_l0.1 | 42.37 / 37.47 | 42.35 / 37.45 | 0.0077 / 0.0192 | 0.0077 / 0.0191 | 232.0 dB | 66.2 dB |
| es_dec_erb_l0.1 | 38.38 / 34.88 | 38.37 / 34.88 | 0.0186 / 0.0350 | 0.0185 / 0.0347 | 232.0 dB | 61.9 dB |

Even-sample match is the ×8 query at the ×4 lattice points against the ×4 query. With held noise it is 232 dB for every arm and utterance — the fp32 floor.

**Figures.** [`figs/scale_agreement.png`](figs/scale_agreement.png) — agreement per arm, all six utterances, τ=0 against τ=1. [`figs/scale_spectrum.png`](figs/scale_spectrum.png) — the ×8 query's spectrum out to 48 kHz for p236_002, showing what sits past the 24 kHz line. [`figs/scale_sweep.png`](figs/scale_sweep.png) — the decoder response over c for each arm, filled dots at the four trained phases, open dots at the four the ×8 query invents.

## 2. Reading

The claim is **confirmed**. On the arm, utterance and τ the note used — `es_erb_l0.1`, p236_002, τ = 0 — step 16000 gives **50.03 dB agreement, 0.0011 % above 24 kHz and a curvature ratio of 0.87**, against 49.90 dB, 0.00 % and 0.88 at step 13500. It holds for all seven arms: the worst arm mean is 38.38 dB, the worst single utterance is 32.93 dB, and no cell leaks more than 0.057 % above 24 kHz. Agreement measures only that leak — across all 78 arm × τ × utterance cells it tracks −(above 24 kHz in dB) at r = 0.999, offset 0.44 dB. At the samples the two lattices share the ×8 query reproduces the ×4 query to 232 dB, and the round trip costs at most 0.115 dB of SNR against the truth anywhere in the table. Decoder noise changes nothing: held and fresh differ by 0.02 dB in agreement and 0.0002 points in leak, and though fresh noise drops the even-sample identity from 232 dB to 66.2 and 61.9 dB, that still sits more than 20 dB above the agreement it has to support. One arm is an outlier — `es_marg` sits 4 to 7 dB under the pack at τ = 0, 40.34 dB mean and 33.37 dB worst, and leaks 0.0173 % against 0.0029 to 0.0048 % for the rest. It is the λ = 0.01 arm, and its ×4 output already carries the most energy at the 24 kHz edge: a band-limited filter ceiling of 49.46 dB where every other arm is 56 to 61. Turning the sampler on costs the ERB arms most, −8.88 dB for `es_dec_erb_l0.1`, −6.70 for `es_erb_l0.1` and −4.11 for `es_dec_l0.1`, against −0.10 for `es_split_l0.1`. The arms that push the most energy to the top of the band have the most to spill past 24 kHz, and even there 0.019 % is −37 dB. The curvature says the decoder did not overfit its four trained coordinate phases: `curv_at` and `curv_gap` both sit at 2–4 × 10⁻⁷ for every arm and every τ, the pooled median ratio is 1.18, and the twelve cells above 3 are denominator artefacts where `curv_at` falls to 3 × 10⁻⁸ while `curv_gap` does not move. Nothing in [`figs/scale_sweep.png`](figs/scale_sweep.png) bumps at c = ±0.25 or ±0.75, on any arm. One wording correction — "monotone" was p236's property, not the run's: 45 of the 78 sweeps turn at least once, though the ones that turn most are the ones whose response barely varies at all (range 0.003 against 0.03 typical), and the median total-variation ratio is 1.009. Query at ×4 or above and decimate: **above ×4 is safe, below ×4 aliases**, and the demo's ×2 path — query ×4, resample down — is backed by every row above.

---

**Recomputed here, beyond the JSON:** the agreement-versus-leakage correlation, the pooled curvature statistics, and the monotonicity and total-variation counts, all read straight out of `scale_OV3_fast_sweeps.npz` and the per-utterance records. No model was run.

**Schema note:** in the JSON summaries, `worst` for `snr_4x_db` and `snr_8x_dec_db` is the maximum, not the minimum (`audit/scale_freedom_all.py:187` only lists the agreement-like keys as min-is-worst). Those two fields are not used in this note; read `min` instead.
