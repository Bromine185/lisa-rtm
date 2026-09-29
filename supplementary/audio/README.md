# Listening examples

Three utterances, 12 kHz → 48 kHz, FLAC (16-bit, lossless). They come from three test speakers of the
NU-Wave 2 / FLowHigh split, which none of our arms, FLowHigh, NU-Wave 2 or AP-BWE saw in training.
AudioSR's training data includes VCTK with no stated speaker split, so it may have seen them.

| folder | utterance | speaker | text |
|---|---|---|---|
| `p360/` | p360_010, 3.1 s | male, American | "People look, but no one ever finds it." |
| `p361/` | p361_009, 4.5 s | female, American | "There is, according to legend, a boiling pot of gold at one end." |
| `p374/` | p374_010, 3.6 s | male, Australian | "People look, but no one ever finds it." |

## Files in each folder

| file | what it is |
|---|---|
| `00_input_12kHz` | the model input: resample_poly 48 → 12 kHz (no content above 6 kHz) |
| `01_naive_upsample` | the input interpolated back to 48 kHz: an empty high band, the SNR ceiling |
| `02_truth` | the 48 kHz recording |
| `03_det_paper_LISA_recipe` | our reimplementation of LISA trained with its released loss (L1, λ = 0): a deficit of about 21.8 dB |
| `04_det` | the same network with a log-magnitude term, λ = 0.01: still a point estimate |
| `05_ours_draw1` … `07_ours_draw3` | three draws of our sampler (`es_dec_erb`, λ = 0.1), same input, different noise |
| `08_ours_waveform_mean_of_16` | the waveform average of 16 draws: the high band loses most of its energy (the paper's "averaging rebuilds the deficit") |
| `09_ours_logmean16` | the per-bin mean of log-magnitude over 16 draws, phase of draw 1 (the LSD-optimal readout) |
| `10_flowhigh` | FLowHigh as released (one Euler step; every seed gives the same output) |
| `11_apbwe` | AP-BWE (deterministic) |
| `12_nuwave2_draw1`, `13_nuwave2_draw2` | NU-Wave 2, two seeds |
| `14_audiosr_draw1`, `15_audiosr_draw2` | AudioSR (speech checkpoint), two seeds |

## Provenance and processing

- **Our model.** Draw 1, the mean of 16 and the log-mean of 16 come from the PyTorch model at τ = 1,
  seed 0. Draws 2 and 3 come from a JavaScript port of the same network, run from the same weights with
  seeds 1 and 2. At τ = 0 that port matches PyTorch to within 1e-4 maximum absolute error.
  - Its noise generator differs from PyTorch's, so draws 2 and 3 are further independent draws, not
    PyTorch seeds 1 and 2.
  - Checked on these files: the draws' low bands correlate at 0.9998–0.9999. Their high bands (above
    6 kHz) correlate at 0.03–0.09 and differ in level by at most 2.2 dB.
- **Raw outputs.** Our outputs are the network's raw outputs: no low-band passthrough and no gain.
- **Released models.** These went through each model's own inference code on the same 12 kHz input (see
  `code/sota/run_models.py`). They were then scaled so that their energy below 5.5 kHz matches the truth's,
  which is the scorer's convention.
- **Different input filter.** These examples use the plain resample_poly input. The tables use the
  Chebyshev-filtered input of the FLowHigh / NU-Wave 2 protocol. The two differ only in the 5.5–6 kHz
  transition band.

## Licence

The recordings are from the CSTR VCTK Corpus, version 0.92 (Yamagishi, Veaux and MacDonald, University of
Edinburgh, 2019), licensed CC BY 4.0. The model outputs are derived from those recordings.
