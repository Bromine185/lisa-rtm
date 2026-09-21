# OV50 evaluation: the conversion, and what the eight arms did

Follow-on to `notes/2026-09-20-eval-handoff.md`. Everything below was run on a laptop, on CPU, from
the checkpoints in `~/lisa-results/`. Nothing was retrained.

## What was run

    fast/convert_ckpt.py     train_arm checkpoints -> the blob e4_eval's load_arm reads, and verify it
    fast/vctk_local.py       the held-out speakers out of the Hub, without the 11.7 GB
    fast/run_e4_eval.py      overnight3/e4_eval.py on EVAL12: LSD, CRPS, PIT
    fast/val_wave_check.py   the converted weights back through their own objective on the run's own
                             validation batches
    fast/run_e5_visqol.py    overnight3/e5_visqol.py, after a separate install

Outputs are in `overnight3/results_OV50/`, copied there by `fast/publish_results.py`. The 48 MB of audio stays
out of the repo and is rebuilt in about three minutes by `fast/run_e4_eval.py`.

## The format gap, confirmed

The handoff called both silent failures. Both are real, and one is worse than it looked.

`load_arm` reads `ck["arm"][0].startswith("det")`. On the string `train_arm.save()` writes, that is
the character `"d"`, and `"d".startswith("det")` is False, so both deterministic arms would have run
at tau = 1. There is a second-order reason this matters more than a wrong number: `det_paper` and
`det` are the reference line the six samplers are judged against. Handicapping the reference
flatters every sampler, and the direction of the error is the direction of the conclusion.

The class default is the other one. `ck.get("cls", "LISAS")` would load the two `es_dec_*` arms as
LISAS. Here it does raise -- `n_dec = 4` makes the decoder's first Linear 101 wide instead of 97 --
but only because n_dec happens to move a shape. The class is metadata and has to travel as metadata.

`fast/convert_ckpt.py` takes `(kind, lambda, cls)` from `ARMS` in `fast/run_contract.py`, the same
table `train_arm.py` trained from, so the name in the filename and the spec in the blob cannot drift.
It checks four things per arm, and all eight arms pass all four:

    params->module   every exported tensor bitwise equal to ck["params"][key][0]   (18/18, all arms)
    forward          stack and module on one batch at fixed noise                  (max|diff| 0.0)
    round-trip       load_arm on the written file: class, tau, weights             (LISAS/LISASD, tau 0/1)
    history          step == 104950 and the checkpoint's own val_wave tail == the history json

The forward check came out at exactly zero rather than at the 1e-5 the gate allows. That is not luck:
with one arm in the stack, the grouped convolution over the arm axis is the same arithmetic in the
same order as the plain one.

An independent confirmation that the det/es split in `ARMS` is the split that trained: the `spread`
term in the history is exactly 0 at every logged step for `det` and `det_paper`, and non-zero for all
six samplers. `tau = 0` for those two is read off the training record, not assumed from the name.

## Rebuilding the validation batches without the 32 GB corpus

`train_arm.py` sets `val_corpus = corpus`, so the 8 x 64 validation segments come out of the full
37.3 h training corpus, which died with the instance. It did not have to be rebuilt.

`stream(VAL_TAG)` draws `idx` from `n` alone and `starts` from `lens[idx]`. A length is a FLAC
header, not a decode. And `off[idx] + starts` always lands a whole segment inside utterance `idx`,
so the concatenation is never needed either -- decoding those utterances and slicing each one gives
the same audio. So: read every shard once for headers, decode 505 utterances, done.

The check that this is the same corpus is arithmetic, not faith. `manifest.json` records
`n = 39639` and `hours = 37.32235759259259`, and the reconstructed index gives

    n = 39639   sum(lens) = 6,449,303,392 samples = 37.32235759259 h

to the sample. One utterance more, one fewer, one in a different place in the sort, and that integer
moves.

## Reproducing val_wave: the answer, and what it cost to get an honest one

`convert_ckpt.py` proves a faithful COPY and nothing more. A converter that paired `es_marg`'s
weights with `es_erb_l0.1`'s name would pass every check in it. Putting the converted weights back
through `val_loss_fast` on the run's own validation batches is the check that catches that, and the
verification the handoff asked for.

Two of the recorded number's inputs do not exist off the training hardware, and neither is a bug.
**AMP**: `FAST["AMP"]` is `_CUDA`, so training ran the encoder and decoder under bf16 autocast with
the distances in fp32; here the whole forward is fp32. **eps**: `val_loss_fast` draws the sampler's
noise from `torch.Generator(device=DEVICE)`, and a CUDA generator and a CPU generator at seed 1234
are different streams -- not recoverable by borrowing a GPU either, since per `train_arm.py`'s header
the CUDA stream is not portable between GPU models. The two deterministic arms take the `is_det`
branch, which never touches eps, so only AMP applies to them.

**The first version of this check asserted a tolerance instead of measuring one, and that was the
mistake worth recording.** In fp32 alone, `es_erb_l0.001`'s `val_spec` lands 8.9% from its own
recorded curve and 3.6% from `es_marg`'s -- so on raw percentages it matches the wrong arm, and a
nearest-match test reports a failure that is not one. Nothing about 8.9% is interpretable until you
know what a different eps stream is worth on that particular arm.

So both terms get measured: four eps seeds for the spread, one bf16 recomputation for the AMP term.
The verdict becomes a prediction -- recorded should equal `mean(fp32 over seeds) + (bf16 - fp32)`,
inside `sd * t(n-1) * sqrt(1 + 1/n)`, the interval for one further draw.

| arm | term | recorded | eps sd | AMP | residual | bands |
|---|---|---|---|---|---|---|
| det | wave | 0.00434071 | -- | +0.48% | -0.071% | 0.14 |
| det | spec | 0.814028 | -- | +0.74% | +0.056% | 0.11 |
| det_paper | wave | 0.00406263 | -- | +0.95% | -0.075% | 0.15 |
| det_paper | spec | 1.60338 | -- | **-17.52%** | -0.026% | 0.05 |
| es_dec_erb_l0.1 | wave | 0.00356711 | 0.085% | +0.52% | -0.334% | 1.11 |
| es_dec_erb_l0.1 | spec | 0.276769 | 0.112% | -0.90% | -0.097% | 0.24 |
| es_dec_l0.01 | wave | 0.00327247 | 0.084% | +0.94% | -0.163% | 0.55 |
| es_dec_l0.01 | spec | 0.433973 | 0.113% | -1.87% | -0.629% | 1.53 |
| es_erb_l0.001 | wave | 0.00321946 | 0.074% | +0.34% | -0.169% | 0.64 |
| es_erb_l0.001 | spec | 0.386113 | 0.094% | **-8.34%** | +0.128% | 0.35 |
| es_erb_l0.01 | wave | 0.00325941 | 0.049% | +0.13% | -0.142% | 0.81 |
| es_erb_l0.01 | spec | 0.282234 | 0.070% | -1.94% | -0.123% | 0.48 |
| es_erb_l0.1 | wave | 0.0035391 | 0.046% | -0.65% | -0.007% | 0.04 |
| es_erb_l0.1 | spec | 0.277234 | 0.115% | -1.80% | -0.057% | 0.14 |
| es_marg | wave | 0.00326217 | 0.028% | +0.41% | -0.028% | 0.28 |
| es_marg | spec | 0.435952 | 0.249% | -4.49% | +0.353% | 0.38 |

Every arm reproduces its recorded curve to under 0.7% once AMP is accounted for, most to under 0.2%.
The AMP column runs from +0.13% to -17.5% depending on the arm, which is exactly why one asserted
tolerance was never going to fit.

**The verdict is identity, and identity passes by 13 to 1800 bands.** Two residuals sit outside the
band, at 1.1x and 1.5x, and the first version of the check failed the whole thing on them. That was
the wrong call: the band models the eps draw, while the AMP correction is CPU bf16 standing in for
A100 bf16 -- different kernels, different accumulation order -- and the error in that stand-in is
not modelled by anything here. The sign test says so rather than leaving it arguable: all eight
`val_wave` residuals come out negative (two-sided p = 0.0078) at 0.1-0.3%, while `val_spec` scatters
5/8 (p = 0.73). The eps draw does not agree in sign eight times. Widening the band until the two fit
would have been fitting the test to the answer, so the claim was split in two instead.

### det_paper's recorded val_spec is AMP noise counted as signal

The largest AMP term in the table, -17.5%, belongs to the quietest arm, and it is worth stating on
its own. `det_paper` is 27.6 dB down in the high band, so its own output there sits around the order
of bf16's per-sample rounding noise on a full-scale waveform, and `lm = |log(H + 1e-7) - log(Y + 1e-7)|`
counts that noise as signal. On the A100 the quantisation partly FILLED the band; in fp32 the same
checkpoint scores 1.944 where the run recorded 1.603.

It changes no conclusion -- lambda = 0 means the spectral term was never in `det_paper`'s loss, so
it is a diagnostic there and not an objective -- but `det_paper`'s recorded `val_spec` should not be
quoted beside the other arms' as though it measured the same thing.

The effect is not simply "quieter arms are more contaminated", which was the first guess and is
wrong. `det` and `es_erb_l0.001` are equally quiet (-12.61 and -12.34 dB) and their AMP terms have
opposite signs (+0.74% and -8.34%). They are not the same functional: the samplers' spectral term is
an energy-score difference, `0.5(dl(Ly,L1) + dl(Ly,L2)) - 0.5 dl(L1,L2)`, so bf16 noise inflates the
SUBTRACTED spread term and pushes the score down, while the `is_det` branch is `sc + lm` with nothing
subtracted. Within each family quietness drives the effect monotonically; the families start from
different baselines.

### The corpus, rebuilt without the 32 GB

What the rebuild settled, and it was the part in doubt: the validation batches are the run's own.
`manifest.json` records `n = 39639` and `hours = 37.32235759259259`; the reconstructed index gives
`n = 39639` and `sum(lens) = 6,449,303,392` samples, that figure to the sample. One utterance more,
one fewer, or one in a different place in the sort, and that integer moves.

There is also a real property of `ArmStack.losses` found by splitting the batches for memory. Every
term is a per-sample mean over the batch axis -- which averages correctly over equal chunks -- EXCEPT
the spectral-convergence term in the `is_det` branch (`e2b_fast.py:347`):

    sc = (Y - H).pow(2).sum((1, 2, 3)).sqrt() / (tf.mag_norm[s] + 1e-8)

That sums over the batch inside the square root and divides by a norm taken over the whole batch, so
it is a ratio of batch aggregates and a mean of per-chunk ratios is a different number. Measured on
`det`, splitting four ways moves `val_spec` by 0.2%. Nothing in the trainer splits batches, so this
is latent rather than a live bug -- but it is worth knowing before anyone does.

## Cross-arm numbers from the training record

`fast/compare.py` against `~/lisa-results`, all eight at step 104950/104950:

| arm | val_wave | val_spec | own objective | kind | lambda | class |
|---|---|---|---|---|---|---|
| es_erb_l0.001 | 0.003219 | 0.386113 | 0.003606 | es_marg_erb | 0.001 | LISAS |
| es_erb_l0.01 | 0.003259 | 0.282234 | 0.006082 | es_marg_erb | 0.01 | LISAS |
| es_marg | 0.003262 | 0.435952 | 0.007622 | es_marg | 0.01 | LISAS |
| es_dec_l0.01 | 0.003272 | 0.433973 | 0.007612 | es_marg | 0.01 | LISASD |
| es_erb_l0.1 | 0.003539 | 0.277234 | 0.031262 | es_marg_erb | 0.1 | LISAS |
| es_dec_erb_l0.1 | 0.003567 | 0.276769 | 0.031244 | es_marg_erb | 0.1 | LISASD |
| det_paper | 0.004063 | 1.603377 | 0.004063 | det | 0 | LISAS |
| det | 0.004341 | 0.814028 | 0.012481 | det | 0.01 | LISAS |

On `val_wave` every sampler sits below both deterministic arms, and `det_paper` sits below `det`.
**Read that column carefully, because it is not what it looks like.** `ArmStack.losses` returns
plain `|yh - y|` for a deterministic arm and the two-draw energy score
`0.5(|y-y1| + |y-y2|) - 0.5|y1-y2|` for a sampler. Those are one proper score -- a point mass's
energy score is its MAE -- so the comparison is legitimate in exactly the way CRPS is. But a
sampler's number is its single-draw L1 *minus half its spread*, and on the validation batches that
spread is 63-154% of `val_wave`. `fast/val_wave_decompose.py` measures the parts, with the
identities checked at tensor level (9e-10) and the `val_wave` anchored to the recorded curves
(3e-10):

| arm | val_wave | val spread | single-draw L1 | vs det |
|---|---|---|---|---|
| det_paper | 0.004027 | 0 | **0.004027** | -7% |
| det | 0.004323 | 0 | 0.004323 | -- |
| es_dec_erb_l0.1 | 0.003564 | 0.002236 | 0.004682 | +8% |
| es_erb_l0.1 | 0.003561 | 0.002308 | 0.004715 | +9% |
| es_marg | 0.003249 | 0.003971 | 0.005234 | +21% |
| es_dec_l0.01 | 0.003250 | 0.003970 | 0.005235 | +21% |
| es_erb_l0.01 | 0.003262 | 0.004035 | 0.005279 | +22% |
| es_erb_l0.001 | 0.003212 | 0.004939 | **0.005681** | +31% |

On waveform error the ranking reverses end to end: `det_paper` is the best waveform predictor in the
run and `es_erb_l0.001`, first on `val_wave`, is the worst. `det_paper` winning plain L1 is what a
pure L1 objective should do and is the reason it is not a strawman -- and it is the same fact as its
-27.6 dB deficit: emitting nothing above 6 kHz is the L1-optimal answer to an unpredictable band.
The earlier version of this note said "every sampler beats both deterministic arms on the shared
waveform term" and let that stand as a result. It is a proper-score result, with the same caveat as
the CRPS one: the samplers are credited for spread a deterministic arm structurally cannot have.

## EVAL12

`overnight3/results_OV50/table_OV50.md`. Two warnings on the columns, both of which have already
caught someone. LSD is `lsd_db`'s convention -- decades of power, not dB; multiply by 10 for dB, and
do not put it beside LISA's published numbers without saying so. And the **`LSD` column here is RAW,
with no baseband passthrough**, where the 16 Sep note reports every condition WITH passthrough. The
passthrough columns are the last two. Comparing this table's `LSD` against that note's `LSD` compares
different quantities and makes this run look worse than it is.

| arm | SNR | LSD | HB-LSD | deficit dB | CRPS | sliced CRPS | HB kappa | PIT end | mean-16 SNR | mean deficit | SNR gap | 1-draw pt LSD | logmean16 pt LSD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| det | 18.84 | 0.916 | 1.044 | -12.61 | 0.9597 | 0.9536 | +0.087 | - | - | - | - | 0.910 | - |
| det_paper | 19.01 | 2.122 | 2.446 | -27.63 | 2.5576 | 2.1312 | +0.269 | - | - | - | - | 2.122 | - |
| es_dec_erb_l0.1 | 18.22 | 0.950 | 1.069 | -6.85 | 0.5700 | 0.5636 | +0.028 | 0.182 | 18.91 | -15.93 | 0.69 | 0.932 | 0.835 |
| es_dec_l0.01 | 17.95 | 0.982 | 1.095 | -7.05 | 0.6002 | 0.5808 | +0.013 | 0.202 | 18.92 | -17.39 | 0.97 | 0.956 | 0.873 |
| es_erb_l0.001 | 17.83 | 1.052 | 1.143 | -12.34 | 0.7284 | 0.6776 | +0.008 | 0.398 | 18.87 | -19.26 | 1.04 | 0.998 | 0.976 |
| es_erb_l0.01 | 17.95 | 0.972 | 1.085 | -7.02 | 0.5748 | 0.5675 | +0.017 | 0.186 | 18.93 | -17.45 | 0.97 | 0.946 | 0.841 |
| es_erb_l0.1 | 18.07 | 0.953 | 1.073 | -6.26 | 0.5767 | 0.5685 | +0.031 | 0.187 | 18.90 | -15.62 | 0.82 | 0.936 | 0.844 |
| es_marg | 18.01 | 0.999 | 1.111 | -7.68 | 0.6115 | 0.5914 | +0.007 | 0.207 | 18.91 | -17.08 | 0.90 | 0.969 | 0.883 |
| floor: pt + empty HB | 18.99 | 5.606 | 6.470 | -52.65 | - | - | - | - | - | - | - | - | - |
| ceiling: pt + true HB | 41.60 | 0.105 | 0.045 | -0.00 | - | - | - | - | - | - | - | - | - |

SNR is doing nothing here and the floor row says why: passthrough with an EMPTY high band scores
18.99 dB, above six of the eight arms. Every decibel of high band a model puts back costs it SNR.
Read the deficit, LSD and CRPS columns.

### The PIT end-bins are not measuring dispersion

This is the one place the handoff's framing needs correcting. It reads `pit_end` as narrowness --
"the ensemble too narrow". Split the two ends and that is not what happened:

| arm | bin 0 / ideal | bin 16 / ideal | pit_end / ideal |
|---|---|---|---|
| es_dec_erb_l0.1 | 1.15 | 1.94 | 1.55 |
| es_erb_l0.01 | 1.12 | 2.04 | 1.58 |
| es_erb_l0.1 | 1.11 | 2.07 | 1.59 |
| es_dec_l0.01 | 1.00 | 2.44 | 1.72 |
| es_marg | 1.01 | 2.51 | 1.76 |
| es_erb_l0.001 | 1.13 | 5.64 | 3.38 |

The bottom bin is at its ideal value, within 15%, for every arm. All the excess is in the top bin:
the truth's high-band log-magnitude sits ABOVE all sixteen draws two to five times more often than
chance. That is a one-sided energy bias, not a narrow ensemble, and it is the same defect the
deficit column reports from the other side (-6 to -8 dB for one draw, -16 to -19 for the mean).

It changes what to do about it. Widening the ensemble to drive `pit_end` to 0.1176 would push the
bottom bin above ideal while the top bin stayed high. The thing to fix is the bias.

The spread-skill curves say the same thing with more resolution: at the high-spread end the arms sit
almost on the diagonal (spread 1.17, skill 1.19 for `es_erb_l0.1`), and at the low-spread end they
are off it by a factor of two (0.41 against 0.83). The ensemble is calibrated where it is unsure and
overconfident where it is confident.

### The questions the run was built to answer

**Does the ERB term help, at matched lambda?** Yes, and on every axis. `es_marg` against
`es_erb_l0.01`, same kind family, same lambda, same class, same batches:

    LSD      0.999 -> 0.972        CRPS     0.6115 -> 0.5748   (-6.0%)
    PIT end  0.2070 -> 0.1859      deficit  -7.68 -> -7.02 dB
    corr_err 0.900 -> 0.705        top bin  2.51x -> 2.04x ideal

`corr_err` is the Frobenius distance between the truth's and the prediction's high-band group
correlation matrices. A 22% drop is the largest single effect in the run, and it is the one an ERB
term is supposed to produce: the bands stop being predicted independently.

**Does lambda = 0.1 overshoot into over-dispersion?** No, and nothing overshoots. Along the
`es_marg_erb` ladder:

    lambda    LSD     CRPS     PIT end   deficit   corr_err
    0.001    1.052   0.7284    0.3978    -12.34     0.733
    0.01     0.972   0.5748    0.1859     -7.02     0.705
    0.1      0.953   0.5767    0.1873     -6.26     0.910

CRPS and PIT are flat between 0.01 and 0.1 -- 0.5748 against 0.5767, 0.1859 against 0.1873. LSD and
deficit still improve at 0.1, and `corr_err` gets 29% WORSE. So lambda = 0.1 buys high-band energy
by spending correlation structure, and the sampler's calibration does not move either way. This is a
change from OV3_fast, where lambda = 0.1 was the winner outright; at 50 epochs it wins only on the
energy columns. If one number has to be picked, lambda = 0.01 is the better point.

lambda = 0.001 is not a weak version of the others, it is a different animal: PIT end 3.38x ideal,
top bin 5.64x, deficit -12.34, next to deterministic `det`'s -12.61. At that
weight the spectral term cannot pay for the spread the energy score would otherwise buy, and the arm
collapses back toward a point predictor. Its val_wave is the best in the run and its single-draw waveform L1 is the worst (0.005681, 31% above `det`): the largest spread in the run buys it the largest credit on a proper score while it reconstructs the waveform worse than any other arm.

**Does decoder noise help?** Yes, small and consistent. Both matched `(kind, lambda)` pairs move the
same way, which is what the paired batch stream was built to detect:

    es_marg    -> es_dec_l0.01       CRPS -0.0113   PIT -0.0049   corr_err -0.023   LSD -0.017
    es_erb_l0.1 -> es_dec_erb_l0.1   CRPS -0.0067   PIT -0.0054   corr_err -0.031   LSD -0.003

Four channels of noise per output sample cost 576 weights out of 88,353 and buy about a fifth of what
the ERB term buys. `es_dec_erb_l0.1` -- both changes together -- is the best arm on CRPS (0.5700),
PIT (0.182) and the logmean16 readout (0.835), which is the strongest single condition in the run.

**Where the deterministic arms still win.** `det`'s one-draw LSD is 0.916, better than every
sampler's. It buys that by staying quiet: its deficit is -12.61 dB against `es_erb_l0.1`'s -6.26. LSD
punishes error in both directions, and a model that under-shoots the high band by 12 dB is wrong by
less, in log-magnitude, than one that gets the level roughly right with the wrong fine structure. The
samplers' case is CRPS (0.57 against 0.96, a 41% drop), the deficit, and the readout column, not LSD
on a single draw.

`det_paper` -- LISA's released configuration, lambda = 0 -- is where the whole argument starts: LSD
2.122, deficit -27.63 dB, CRPS 2.56. It also has the best SNR in the table (19.01) and the highest
high-band kappa (+0.269). It is behaving as a conditional mean: almost no output energy, and what
little there is correlates with the truth.

## Against the 16 Sep run: better, but not where it was supposed to be

OV3_fast was 7.6 epochs; OV50 is 50. Like-for-like, one draw, both with passthrough:

| arm | 16 Sep | OV50 | |
|---|---|---|---|
| `det` | 1.198 | 0.910 | -24% |
| `es_erb_l0.1` | 0.972 | 0.936 | -3.7% |
| `es_dec_erb_l0.1` | 0.999 | 0.932 | -6.7% |
| `es_marg` | 0.946 | 0.969 | **+2.4% worse** |

And `logmean16` + passthrough, the readout the 16 Sep note argued for: `es_erb_l0.1` 0.871 -> 0.844,
`es_marg` 0.920 -> 0.883, `es_dec_erb_l0.1` 0.933 -> 0.835. Seven of eight figures improved.
(`es_dec_l0.1` is not comparable: 16 Sep ran it at lambda = 0.1, OV50 at 0.01.)

**The floor and ceiling rows are identical to three decimals across the two runs** -- LSD 5.606 and
0.105, deficit -52.65 and -0.00, SNR 18.99 and 41.60. Those conditions are model-independent, so
that is proof rather than coincidence that `fast/vctk_local.py`'s partial read reproduces exactly the
EVAL12 the 16 Sep run used. The data path is verified by result, not only by construction.

**But `det` gained far more than any sampler**, and that is the finding, not the LSD numbers. There
is a mechanism for it. A single draw's LSD has an irreducible floor: even a perfectly learned
conditional law produces draws that deviate from the conditional mean by the true conditional
variance, and training does not remove it. `det` has no such floor and can keep walking toward the
conditional-mean optimum. So with more epochs `det`'s single-draw LSD should keep improving while
the samplers' saturates, and their `logmean16` should keep improving because it estimates the mean.
Both runs show exactly that.

## ViSQOL

`overnight3/results_OV50/visqol_table_OV50.md`, 43 conditions, M = 16, lattice speech mapping (so
the speech numbers are OV2-comparable; `ai_edge_litert` has a cp311 arm64 wheel and the run confirms
the mapping is live). Audio mode at 48 kHz sees the whole 6-24 kHz band; speech mode resamples to
16 kHz and sees about 11% of it. Lead with audio.

Audio-mode floor 1.567, ceiling 4.725, range 3.158.

| condition | audio48k | share of range | LSD |
|---|---|---|---|
| ceiling: passthrough + true HB | 4.725 | 100% | 0.105 |
| `es_dec_erb_l0.1` logmean16 + pt | **3.021** | **46.0%** | 0.835 |
| `det` + pt | 3.001 | 45.4% | 0.910 |
| `es_erb_l0.1` logmean16 + pt | 2.991 | 45.1% | 0.844 |
| `es_erb_l0.01` logmean16 + pt | 2.942 | 43.5% | 0.841 |
| `es_dec_l0.01` logmean16 + pt | 2.927 | 43.1% | 0.873 |
| `es_marg` logmean16 + pt | 2.877 | 41.5% | 0.883 |
| `es_dec_erb_l0.1` one draw + pt | 2.815 | 39.5% | 0.932 |
| `det_paper` + pt | 1.717 | 4.7% | 2.122 |
| naive | 1.576 | 0.3% | 4.776 |
| floor: passthrough + empty HB | 1.567 | 0% | 5.606 |

**The sampler's advantage on the judge that can see the band has essentially gone.** On 16 Sep the
best sampler readout beat `det` by 0.290 audio ViSQOL (2.773 against 2.483, 38.2% of range against
29.0%). At 50 epochs it is 0.020 (3.021 against 3.001, 46.0% against 45.4%) -- six tenths of one
percent of the range. `det` gained +0.518 over the two runs; the best sampler gained +0.248.

That is the same saturation the LSD comparison shows, on an independent judge, and it is the
strongest evidence in the run against the sampler programme as stated.

### PESQ ranks doing nothing second

| condition | PESQ wb | LSD | audio48k |
|---|---|---|---|
| ceiling | 4.615 | 0.105 | 4.725 |
| **`det_paper` + pt** | **4.385** | 2.122 | 1.717 |
| **naive** | **4.377** | 4.776 | 1.576 |
| `es_marg` mean16 + pt | 4.354 | 1.193 | 2.576 |

`det_paper` is 27.6 dB down in the high band and `naive` has no high band at all, and PESQ puts them
second and third of 43 conditions, above every sampler. Speech-mode ViSQOL is milder but points the
same way: its whole range for this task is 3.947 to 4.508, and `naive` scores 3.972, above six
conditions that actually restore energy. Both judges should be retired for 12 -> 48 kHz, which the
17 Sep TODO already proposed; this is the evidence for it.

## Does the sampler programme clear its own bar?

The rule set on 16 Sep was that a winner must improve **CRPS and calibration and LSD and audio-mode
ViSQOL together**. At 50 epochs:

- **CRPS** -- yes, 0.5700 against `det`'s 0.9597. But a point forecast is a degenerate predictive
  distribution and must lose a proper probabilistic score, so this clause is partly definitional and
  should not carry the argument.
- **LSD** -- yes, at the right readout: 0.835 against 0.910, and five of six samplers beat `det`.
- **audio ViSQOL** -- 3.021 against 3.001. Technically yes; 0.6% of the range, which is not a result.
- **calibration** -- no. PIT end-bins 1.55x ideal at best against 1.79x in OV3_fast, with all the
  excess in the top bin. The SNR gap is 0.69-1.04 dB against the 2.75 dB a calibrated 16-member
  ensemble would give: every arm carries a quarter to a third of the spread it should. The
  draw/logmean16 ratio puts the conditional variance at about a quarter of the squared bias.

So: the samplers halve the high-band deficit against `det` (-6.26 dB against -12.61) and beat it at
the right readout, and that part is real and reproducible. But they are conditional PREDICTORS with
some usable spread, not calibrated conditional DISTRIBUTIONS, and 6.5x more training than OV3_fast
did not move the calibration and let `det` close the perceptual gap to nothing.

**What would change that verdict.** The top-bin excess is an energy bias, not a width problem -- the
bottom bin is at its ideal value for every arm. Nobody has tried simply correcting it. If a per-band
gain fitted on held-out data moves `pit_end` toward 0.118, then the ensembles were mis-calibrated in
level and not in shape, which is a different and much more tractable claim. That test needs no
retraining.

## Not done

**FAD.** Still not in the repo, and still a study-design question before it is a coding one. EVAL12
is twelve clips; FAD is unstable below a few hundred.

**The gated deficit** (TODO, 17 Sep). The high band does not gate with the speech, and the mean
deficit hides it. Every number in the deficit column above is a mean over frames.
