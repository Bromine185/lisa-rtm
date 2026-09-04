# Research log -- XL_relu_l1e-3

LISA reimplementation, 86,881 params (paper ~89k), 12000 Hz -> 48000 Hz,
97 train / 3 held-out VCTK speakers, seed 0.

## Gate 1 -- closed-form validation
All 12 tests pass: T1 reproduces the analytic log-Rayleigh shift map to 1e-12,
Bures satisfies A.Cs.A = Ct to 1e-10, McCann endpoints are exact and the W2 geodesic identity holds.

## Gate 2a -- fidelity
Waveform SNR on held-out speakers: **19.00 dB** (naive upsampling 19.01 dB; paper 24.16 dB).
LSD 1.708 in this notebook's basis, 1.695 in the paper's STFT basis (not like-for-like; see §9).
Verdict: FAIL -- below naive upsampling; nothing below is a result.

## Gate 2b -- headroom
Mean high-band energy deficit on held-out speakers: **-23.46 dB**.
Verdict: headroom exists.

## Results
| condition | SNR dB | LSD | HB-LSD | HB deficit dB |
|---|---|---|---|---|
| T1 quantile (lam=1) | 18.20 | 1.000 | 1.144 | -7.18 |
| T1 quantile (best lam=1.00) | 18.20 | 1.000 | 1.144 | -7.18 |
| S stochastic (1 draw) | 18.42 | 1.021 | 1.169 | -8.04 |
| T2 block (lam=1) | 18.56 | 1.048 | 1.200 | -8.82 |
| T2 block (best lam=1.00) | 18.56 | 1.048 | 1.200 | -8.82 |
| T2 diag (lam=1) | 18.57 | 1.050 | 1.203 | -8.91 |
| T2 diag (best lam=1.00) | 18.57 | 1.050 | 1.203 | -8.91 |
| T3 conditional (lam=1) | 18.43 | 1.055 | 1.208 | -8.16 |
| T3 conditional (best lam=1.00) | 18.43 | 1.055 | 1.208 | -8.16 |
| control: mis-specified T1 | 17.98 | 1.116 | 1.279 | -10.08 |
| control: frame shuffle | 18.58 | 1.315 | 1.509 | -10.44 |
| control: shaped noise | 17.92 | 1.449 | 1.665 | -7.05 |
| T0 identity | 19.00 | 1.708 | 1.965 | -23.41 |

## Findings

- Best rung: **T1 quantile (lam=1)** at HB-LSD 1.144 vs baseline 1.965,
  costing -0.79 dB SNR. The SNR cost is expected and required: the
  missing conditional variance has to come from somewhere.
- **Shaped-noise control**: HB-LSD 1.665. The transport ladder
  beats 1970s noise filling.
  
- **Frame-shuffle control**: HB-LSD 1.509.
  The metric suite can see conditional structure
  
- **Mis-specified map**: HB-LSD 1.279.
  WARNING: a deliberately wrong map still improves HB-LSD, so LSD is rewarding generic energy inflation and must be demoted from primary evidence.
- **CRPS**: deterministic 1.1263 -> stochastic 0.9060.
  Only the sampler can move this; it is the metric on which the generative claim stands or falls.
- **Latency**: transport 32.90 ms per second of audio (p95 35.18), i.e.
  30x real time, 0x cheaper than LISA itself.
  There is no latency case for a C++/Rust rewrite of the correction.

## Still open

- Perceptual adjudication. No metric here settles "sharper" vs "sounds better"; §14 is by ear only.
- Post-hoc spectral transport is classical bandwidth extension in OT vocabulary. The differentiated
  claim would be transport on the 32-d latent codes, where a FULL covariance is 528 entries and is
  estimable -- exactly where it is noise-dominated across 769 STFT bins.
- LISA's decoder is a ReLU MLP, not a SIREN. ReLU spectral bias may be a second, architectural
  cause of over-smoothing, independent of the loss. One sweep of the decoder's frequency response
  would separate the two.
