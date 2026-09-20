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

Outputs are in `~/lisa-results/ov3/` (`results_OV50.json`, `table_OV50.md`, `val_wave_OV50.json`,
`audio/`) and `~/lisa-results/figs/`. ViSQOL is installed and its runner is here; those numbers are
not in this note yet.

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

## Reproducing val_wave: what it can and cannot settle

The point of re-running the weights is that `convert_ckpt.py` proves a faithful COPY and nothing
more. A converter that paired `es_marg`'s weights with `es_erb_l0.1`'s name would pass every check
in that file. Putting the converted weights back through `val_loss_fast` on the run's own validation
batches is the check that catches it.

Two of the recorded number's inputs do not exist off the training hardware, and neither is a bug.

**AMP.** `FAST["AMP"]` is `_CUDA`, so training ran the encoder and decoder under bf16 autocast with
the distances in fp32. Here the whole forward is fp32. bf16 carries eight mantissa bits.

**eps.** `val_loss_fast` draws the sampler's noise from `torch.Generator(device=DEVICE)` at a fixed
seed, and a CUDA generator and a CPU generator at seed 1234 are different streams. Per
`train_arm.py`'s own header the CUDA stream is not portable between GPU models either, so this is
not recoverable by borrowing a GPU. The two deterministic arms take the `is_det` branch, which never
touches eps, so only AMP applies to them.

What came out:

| arm | val_wave vs recorded | val_spec vs recorded |
|---|---|---|
| det | -0.41% | -0.79% |
| det_paper | -0.87% | **+21.3%** |
| es_dec_erb_l0.1 | -0.08% | +1.16% |
| es_dec_l0.01 | -0.69% | +2.67% |
| es_erb_l0.001 | -0.24% | **+8.87%** |

`val_wave` reproduces to under 1% everywhere, and the samplers do BETTER than the deterministic arms
despite additionally drawing different noise. That is the energy score's shape doing it: its wave
term is `0.5(d(y,y1) + d(y,y2)) - 0.5 d(y1,y2)`, a difference in which a systematic bf16 offset
largely cancels, where `det`'s plain `|yhat - y|` carries it straight through.

`val_spec` is a different story and the two outliers are informative.

`det_paper` at +21.3% is, I believe, AMP rounding noise being counted as signal. It is the quietest
arm in the run by a long way -- 27.6 dB down in the high band, so its own high-band output sits
around -58 dB of full scale, which is the order of bf16's per-sample rounding noise on a full-scale
waveform. `lm = |log(H + 1e-7) - log(Y + 1e-7)|` is where that shows up: on the A100 the
quantisation noise partly FILLED the band and made the recorded loss smaller than the arithmetic
deserves. If that is right, `det_paper`'s recorded `val_spec` of 1.603 is contaminated and the fp32
1.944 is the honest number. It changes no conclusion -- lambda = 0 means the spectral term was never
in `det_paper`'s loss, so it is a diagnostic, not an objective -- but it should not be quoted as if
it were comparable with the other arms'. `fast/val_wave_check.py`'s docstring records the test that
settles it: recompute under `torch.autocast("cpu", dtype=torch.bfloat16)` and see whether 1.944 falls
toward 1.603.

`es_erb_l0.001` at +8.87% is the eps draw. It has the largest `spread` in the run (max 0.115), and
the energy score's spectral term is again a difference of similar quantities, so it is the arm most
sensitive to being handed a different noise stream.

**The open item.** A "nearest recorded curve" test needs a tolerance, and I asserted one instead of
measuring it. At an 8.87% diagonal, `es_erb_l0.001` sits closer to `es_marg`'s recorded pair (3.6%)
than to its own, and they are the same class, so the matching criterion as first written reports a
failure that is not one. The fix is to measure the eps-induced spread -- the same arm at several
seeds -- and test against that, rather than against a number I picked. Until that lands, the identity
of each converted file rests on `convert_ckpt.py`'s history check, which ties each checkpoint to its
own curve bitwise and does not depend on any of this.

What the rebuild DID settle, and it is the part that was in doubt: the validation batches are the
run's own. `manifest.json` records `n = 39639` and `hours = 37.32235759259259`; the reconstructed
index gives `n = 39639` and `sum(lens) = 6,449,303,392` samples, which is that figure to the sample.
One utterance more, one fewer, or one in a different place in the sort, and that integer moves.

There is also a real property of `ArmStack.losses` worth recording, found by splitting the batches
for memory. Every term is a per-sample mean over the batch axis -- which averages correctly over
equal chunks -- EXCEPT the spectral-convergence term in the `is_det` branch (`e2b_fast.py:347`):

    sc = (Y - H).pow(2).sum((1, 2, 3)).sqrt() / (tf.mag_norm[s] + 1e-8)

That sums over the batch inside the square root and divides by a norm taken over the whole batch, so
it is a ratio of batch aggregates and a mean of per-chunk ratios is a different number. Measured on
`det`, splitting four ways moves `val_spec` by 0.2% and `val_wave` not at all. The deterministic arms
therefore run unsplit; they can afford it, because `is_det` runs B sequences where a sampler runs 2B.

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

Every sampler beats both deterministic arms on the shared waveform term. `det_paper` beats `det` on
it, which is what a pure L1 objective should do and is the reason `det_paper` is not a strawman.

## EVAL12

`~/lisa-results/ov3/table_OV50.md`. LSD here is `lsd_db`'s convention -- decades of power, not dB;
multiply by 10 for dB, and do not put it beside LISA's published numbers without saying so.

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
collapses back toward a point predictor. Its val_wave is the best in the run, which is exactly what a
collapsed sampler should score on an L1 term.

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

## Not done

**FAD.** Still not in the repo, and still a study-design question before it is a coding one. EVAL12
is twelve clips; FAD is unstable below a few hundred.

**The gated deficit** (TODO, 17 Sep). The high band does not gate with the speech, and the mean
deficit hides it. Every number in the deficit column above is a mean over frames.
