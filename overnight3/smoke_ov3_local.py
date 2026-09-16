"""CPU smoke test of overnight3 (e1 model + distances, e2 trainer, e4 eval pieces) at SMOKE dims (16 kHz).
Boots the notebook's definition cells as overnight2/analysis_local.py does, exec's the XL trainer helpers,
c1_model, e1, e2, builds a synthetic corpus with validate.py's generator, trains all eight e3 arms for 40 steps,
reloads the checkpoints, and runs e4's ensemble metrics + readouts on 2 utterances with M=2.  No network,
no Drive.  Scratch goes to a temp dir.

    venv/bin/python overnight3/smoke_ov3_local.py
"""
import json, os, sys, pathlib, time, tempfile, math
import numpy as np
os.environ.setdefault("MPLBACKEND", "Agg")
REPO = pathlib.Path(__file__).resolve().parents[1]
OV3 = REPO / "overnight3"
SCR = pathlib.Path(tempfile.mkdtemp(prefix="ov3_smoke_"))
os.chdir(SCR)                                     # the notebook's setup cell creates ./lisa_rtm_cache in cwd
T0 = time.time()

NB = json.loads((REPO / "lisa_rtm.ipynb").read_text())
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
G = {"__name__": "__main__"}
def run(mark):
    exec(compile(next(s for s in CODE if mark in s), f"<nb {mark[:20]}>", "exec"), G)
run("import importlib, subprocess, sys")
import torch
G["DEVICE"] = torch.device("cpu")
run("@dataclasses.dataclass(frozen=True)"); G["CFG"] = G["SMOKE"]
run("def _hann(n):"); run("class QuantileMap:"); run("def snr_db(y, y_hat):"); run("VCTK_URL = ")
run("class LISAEncoder(nn.Module):")
_xl = open(REPO / "overnight/cell2_trainer.py").read()
exec(_xl.split("def train_paired")[0], G)                                  # GPUCorpus, naive_upsample, hb_deficit_of
exec("def save_ckpt" + _xl.split("def save_ckpt")[1], G)
exec(open(REPO / "overnight/cell3_eval.py").read(), G)
torch.use_deterministic_algorithms(False)
torch.set_num_threads(max(1, os.cpu_count() // 2))
G["ROOT"] = SCR; G["CKPT"] = SCR / "ckpt"; G["FIGS"] = SCR / "figs"
for d in (G["CKPT"], G["FIGS"]):
    d.mkdir(parents=True, exist_ok=True)
exec(compile(open(REPO / "overnight2/c1_model.py").read(), "<c1>", "exec"), G)
exec(compile(open(OV3 / "e1_model.py").read(), "<e1>", "exec"), G)
exec(compile(open(OV3 / "e2_trainer.py").read(), "<e2>", "exec"), G)
CFG = G["CFG"]
print(f"boot ok: CFG {CFG.name} fs {CFG.fs_lo}->{CFG.fs_hi}  [{time.time()-T0:.0f}s]", flush=True)

# ---- synthetic corpus: validate.py's generator, verbatim -------------------------------------------------
_v = open(REPO / "validate.py").read()
_gen = "def synthetic_corpus" + _v.split("def synthetic_corpus")[1].split("\nfailures = []")[0]
_ns = {"G": G}
exec(_gen, _ns)
_ns["synthetic_corpus"](None)
train_utts, test_utts = G["train_utts"], G["test_utts"]
G["paper_test_utts"] = test_utts
SEG, BATCH = 4096, 4
corpus = G["HostCorpus"](train_utts, CFG, seg_hi=SEG)
val_corpus = G["HostCorpus"](test_utts, CFG, seg_hi=SEG)

# ---- e1 unit checks -------------------------------------------------------------------------------------
LISAS, LISASD, copy_shared = G["LISAS"], G["LISASD"], G["copy_shared"]
m0 = LISAS(CFG).eval(); md = copy_shared(m0, LISASD(CFG)).eval()
n_sd = sum(p.numel() for p in md.parameters()); n_s = sum(p.numel() for p in m0.parameters())
print(f"LISASD params {n_sd:,}  LISAS params {n_s:,}  (+{n_sd - n_s})", flush=True)
assert n_sd - n_s == G["N_DEC"] * CFG.dec_hidden
x, y = corpus.batch(G["stream"]("smoke/x"), 2)
with torch.no_grad():
    a, b = m0(x), md(x)
    assert a.shape == b.shape == y.shape, (a.shape, b.shape, y.shape)
    d0 = float((a - b).abs().max())
    assert d0 < 1e-6, f"LISASD(eps=0) != LISAS(eps=0): {d0}"
    e = md.sample_eps(x, 1.0, 0)
    assert isinstance(e, tuple) and e[0].shape == (2, md.n_noise, x.shape[1]) and e[1].shape == (2, x.shape[1] * CFG.upsample, md.n_dec)
    d1 = float((md(x, eps=e) - b).abs().max())
    assert d1 > 1e-6, "decoder noise has no effect"
    # chunked decode == whole decode with the same eps
    z = md.encode(x, e); full = md.decode(z); parts = torch.cat([md.decode(z, s, min(s + 3000, full.shape[1])) for s in range(0, full.shape[1], 3000)], 1)
    assert float((full - parts).abs().max()) < 1e-5
print(f"LISASD(eps=0) == LISAS(eps=0): max diff {d0:.2e}; decoder noise effect {d1:.3e}; chunked decode ok", flush=True)
for n_fft in (CFG.n_fft, CFG.n_fft // 2, CFG.n_fft // 4):
    W = G["erb_filterbank"](n_fft, CFG.fs_hi).numpy()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / CFG.fs_hi)
    assert W.shape == (32, n_fft // 2 + 1)
    assert np.allclose(W.sum(1), 1.0, atol=1e-6), "ERB rows do not sum to 1"
    assert (W.sum(0)[freqs >= 50.0] > 0).all(), "ERB: uncovered bin above f_lo"
print("erb_filterbank: rows sum to 1, every bin >= 50 Hz covered, at all three scales", flush=True)
spec_loss = G["MultiScaleSTFTLoss"](CFG.n_fft)
for kind in ("es_ged", "es_erb", "es_marg_erb", "es_marg", "det"):
    for cls in (LISAS, LISASD):
        m = cls(CFG); loss, terms = G["arm_loss3"](kind, 1e-2, m, x, y, spec_loss); loss.backward()
        assert math.isfinite(loss.item()) and all(math.isfinite(v) for v in terms.values()), (kind, cls.__name__, terms)
print(f"arm_loss3 finite for every kind x class  [{time.time()-T0:.0f}s]", flush=True)

# ---- train all eight e3 arms for 40 steps -------------------------------------------------------------------
ARMS = {"det": ("det", 1e-2, "LISAS"), "es_marg": ("es_marg", 1e-2, "LISAS"), "es_marg_l0.1": ("es_marg", 1e-1, "LISAS"),
        "es_split_l0.1": ("es_split_marg", 1e-1, "LISAS"), "es_erb_l0.1": ("es_marg_erb", 1e-1, "LISAS"),
        "es_dec": ("es_marg", 1e-1, "LISASD"), "es_dec_erb_l0.1": ("es_split_marg_erb", 1e-1, "LISASD"), "es_ged": ("es_ged", 1e-2, "LISAS")}
TAG = "SMOKE_OV3"
probe = test_utts[0]
G["time_ov3"](corpus, ARMS, BATCH, n=2)
models, hist = G["train_ov3"](corpus, val_corpus, ARMS, 40, BATCH, 1e-3, (0.5,), 0.5, 1e-3, 20, TAG, probe, val_every=10, n_val_batches=8)
for k, h in hist.items():
    for key in ("wave", "spec", "val_loss", "val_wave", "val_spec", "train_loss_ema", "snr0", "def0"):
        assert h[key] and all(math.isfinite(v) for v in h[key]), (k, key, h[key])
    assert h["val_step"] == [10, 20, 30, 40], h["val_step"]
    if ARMS[k][0].startswith("det"):
        assert all(v is None for v in h["snr1"] + h["def1"]), (k, h["snr1"])
    assert h["dev_step"] == [20, 40], h["dev_step"]
    if not ARMS[k][0].startswith("det"):
        assert all(math.isfinite(v) for v in h["snr1"] + h["def1"])
print("history: every loss finite; val_step", hist["det"]["val_step"], "dev_step", hist["det"]["dev_step"], flush=True)
assert (SCR / f"ov3_history_{TAG}.json").exists() and (G["FIGS"] / f"ov3_curves_{TAG}.png").exists()
print("history JSON and curves PNG exist", flush=True)

# ---- reload + full-utterance reconstruct for both classes -----------------------------------------------------
for k in ("es_erb_l0.1", "es_dec_erb_l0.1", "det"):
    m2, ck = G["load_arm"](G["CKPT"] / TAG / f"{k}.pt")
    assert type(m2).__name__ == ARMS[k][2] == ck["cls"], (type(m2).__name__, ck["cls"])
    yy = np.asarray(test_utts[1], np.float64)
    r1 = G["reconstruct"](m2, yy, CFG, chunk=8192, tau=1.0, seed=1)
    r1b = G["reconstruct"](m2, yy, CFG, chunk=8192, tau=1.0, seed=1)
    r0 = G["reconstruct"](m2, yy, CFG, chunk=8192, tau=0.0)
    assert len(r1) == len(yy) and np.isfinite(r1).all() and np.abs(r1 - r1b).max() == 0.0
    print(f"reload {k:<10} {type(m2).__name__:<7} step {ck['step']} tau {m2.tau:g}: SNR tau=1 {G['snr_db'](yy, r1):.2f} tau=0 {G['snr_db'](yy, r0):.2f}"
          f"  tau1-vs-tau0 max diff {np.abs(r1 - r0).max():.3e}", flush=True)

# ---- e4 pieces on 2 utterances, M=2 ------------------------------------------------------------------------------
G["OV3_RUN_EVAL"] = False; G["OV3_TAG"] = TAG
exec(compile(open(OV3 / "e4_eval.py").read(), "<e4>", "exec"), G)
models3, arms3 = G["load_ov3"](TAG)
assert set(models3) == set(ARMS)
RES, TABLE = G["run_eval"](models3, arms3, test_utts[:2], 2, TAG)
for k in ARMS:
    r = RES[k]["eval12"]
    assert math.isfinite(r["crps"]) and math.isfinite(r["single"]["lsd"]) and math.isfinite(r["single_pt"]["lsd"])
    if models3[k].tau > 0:
        assert set(r["readouts"]) == {"logmean2"} and math.isfinite(r["readouts_pt"]["logmean2"]["lsd"])   # logmean4 only when M > 4
        assert len(r["pit_hist"]) == 3
draws = np.stack([G["reconstruct"](models3["es_dec"], test_utts[0], CFG, tau=1.0, seed=s) for s in range(2)])
ro = G["logmag_ensemble_readout"](draws, test_utts[0], CFG)
assert len(ro) == len(test_utts[0]) and np.isfinite(ro).all()
assert (SCR / "ov3" / f"results_{TAG}.json").exists() and (SCR / "ov3" / f"table_{TAG}.md").exists()
assert (G["FIGS"] / f"ov3_spectrum_{TAG}.png").exists() and (G["FIGS"] / f"ov3_calibration_{TAG}.png").exists()
wavs = sorted(p.name for p in (SCR / "ov3" / "audio").glob("*.wav"))
assert "u0_es_dec_logmean2_pt.wav" in wavs and "u0_det_pt.wav" in wavs and "u0_floor_pt.wav" in wavs, wavs
print(f"e4: results/table/figures written, {len(wavs)} wavs", flush=True)

# ---- e3 / e5 syntax only (they need the GPU corpus and the visqol install) ---------------------------------------
for f in ("e3_launch.py", "e5_visqol.py"):
    compile(open(OV3 / f).read(), f, "exec")
print("e3 / e5 compile", flush=True)
print(f"LISASD params: {n_sd:,}")
print(f"scratch: {SCR}")
print(f"SMOKE OV3 OK  [{time.time()-T0:.0f}s]")
