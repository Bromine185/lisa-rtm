# Research log -- XL_relu_l1e-2

LISA reimplementation, 86,881 params (paper ~89k), 12000 Hz -> 48000 Hz,
97 train / 3 held-out VCTK speakers, seed 0.

## Gate 1 -- closed-form validation
All 12 tests pass: T1 reproduces the analytic log-Rayleigh shift map to 1e-12,
Bures satisfies A.Cs.A = Ct to 1e-10, McCann endpoints are exact and the W2 geodesic identity holds.

## Gate 2a -- fidelity
Waveform SNR on held-out speakers: **18.85 dB** (naive upsampling 19.01 dB; paper 24.16 dB).
LSD 0.930 in this notebook's basis, 0.920 in the paper's STFT basis (not like-for-like; see §9).
Verdict: FAIL -- below naive upsampling; nothing below is a result.

## Gate 2b -- headroom
Mean high-band energy deficit on held-out speakers: **-12.89 dB**.
Verdict: headroom exists.

## Results
| condition | SNR dB | LSD | HB-LSD | HB deficit dB |
|---|---|---|---|---|
| T1 quantile (best lam=0.90) | 18.45 | 0.915 | 1.039 | -8.47 |
| T1 quantile (lam=1) | 18.34 | 0.915 | 1.039 | -7.85 |
| T2 diag (lam=1) | 18.72 | 0.917 | 1.041 | -11.14 |
| T2 diag (best lam=1.00) | 18.72 | 0.917 | 1.041 | -11.14 |
| S stochastic (1 draw) | 18.55 | 0.917 | 1.042 | -9.35 |
| T3 conditional (best lam=0.80) | 18.66 | 0.920 | 1.045 | -10.43 |
| T3 conditional (lam=1) | 18.56 | 0.921 | 1.046 | -9.67 |
| T2 block (best lam=0.60) | 18.79 | 0.925 | 1.051 | -11.91 |
| T2 block (lam=1) | 18.71 | 0.927 | 1.053 | -10.96 |
| T0 identity | 18.85 | 0.930 | 1.057 | -13.16 |
| control: mis-specified T1 | 18.10 | 1.024 | 1.167 | -10.37 |
| control: frame shuffle | 18.64 | 1.304 | 1.492 | -11.07 |
| control: shaped noise | 17.82 | 1.353 | 1.549 | -5.88 |

## Findings

- Best rung: **T1 quantile (best lam=0.90)** at HB-LSD 1.039 vs baseline 1.057,
  costing -0.41 dB SNR. The SNR cost is expected and required: the
  missing conditional variance has to come from somewhere.
- **Shaped-noise control**: HB-LSD 1.549. The transport ladder
  beats 1970s noise filling.
  
- **Frame-shuffle control**: HB-LSD 1.492.
  The metric suite can see conditional structure
  
- **Mis-specified map**: HB-LSD 1.167.
  A deliberately wrong map does not improve HB-LSD, so the gain is not generic energy inflation.
- **CRPS**: deterministic 0.9572 -> stochastic 0.7533.
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
