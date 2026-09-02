# ---- Evaluation audit: paste as ONE Colab cell after the §7-§9 definitions exist ----------
# Reuses reconstruct / decimate / band_energy_ratio / snr_db verbatim. Adds nothing to the pipeline.
import numpy as np, torch, matplotlib.pyplot as plt

# Reload both splits from the raw caches -- an earlier cell may have set test_utts = train_utts[:12].
with np.load(FIXTURES / "train_FULL.npz", allow_pickle=True) as d:
    train_utts, train_speakers = list(d["utts"]), d["speakers"]
with np.load(FIXTURES / "test_FULL.npz", allow_pickle=True) as d:
    test_utts, test_speakers = list(d["utts"]), d["speakers"]
print("train:", len(train_utts), np.unique(train_speakers)); print("test: ", len(test_utts), np.unique(test_speakers))

# 0. Are we evaluating the trained weights?  (Cheapest possible check; do this FIRST.)
ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
sd_mem, sd_ck = model.state_dict(), ck["model"]
max_dev = max((sd_mem[k].float() - sd_ck[k].float()).abs().max().item() for k in sd_ck)
print(f"checkpoint step {ck['step']}   in-memory vs checkpoint max |dw| = {max_dev:.3e}")
if max_dev > 1e-6:
    print(">>> in-memory model is NOT the checkpoint. Loading it now.")
    model.load_state_dict(sd_ck)
model.eval()

# 1. Does the prediction respond to its input at all?  (Random-init nets are bias-dominated.)
y0 = np.asarray(train_utts[0], np.float64)
p1, p2 = reconstruct(model, y0, CFG), reconstruct(model, 2.0 * y0, CFG)
print(f"input x2  ->  output RMS ratio {np.sqrt(np.mean(p2**2) / np.mean(p1**2)):.3f}  (want ~2.0)")

# 2. Same-code-path comparison, 12 train vs 12 held-out utterances.
def audit(utts, tag, n=CFG.n_eval_utts):
    rows = []
    for y in utts[:n]:
        y = np.asarray(y, np.float64)
        x = decimate(y, CFG.upsample)
        yh = reconstruct(model, y, CFG)
        b = band_energy_ratio(y, yh, CFG.fs_hi, CFG.eval_n_fft, CFG.eval_hop, 200.0, CFG.fs_hi / 2)
        lo, hi = b[:, 0] < CFG.fs_lo / 2, b[:, 0] >= CFG.fs_lo / 2
        rows.append(dict(len_s=len(y) / CFG.fs_hi, dtype=str(y.dtype), ndim=y.ndim,
                         tgt_peak=np.abs(y).max(), tgt_rms=np.sqrt(np.mean(y**2)),
                         in_peak=np.abs(x).max(), in_rms=np.sqrt(np.mean(x**2)),
                         pred_peak=np.abs(yh).max(), pred_rms=np.sqrt(np.mean(yh**2)),
                         snr=snr_db(y, yh), base_db=b[lo, 1].mean(), hb_db=b[hi, 1].mean()))
    keys = ["len_s", "tgt_peak", "tgt_rms", "in_peak", "in_rms", "pred_peak", "pred_rms", "snr", "base_db", "hb_db"]
    print(f"\n{tag}: {len(rows)} utts   (mean / min / max)")
    for k in keys:
        v = np.array([r[k] for r in rows])
        print(f"  {k:9s} {v.mean():9.4f} {v.min():9.4f} {v.max():9.4f}")
    return rows

tr = audit(train_utts, "TRAIN")
te = audit(test_utts,  "HELD-OUT")
print("\nraw cache dtypes/shapes:", train_utts[0].dtype, train_utts[0].shape, "|", test_utts[0].dtype, test_utts[0].shape)

# 3. One pair each, identical axes.
fig, ax = plt.subplots(2, 2, figsize=(12, 6), sharex="col", sharey=True)
for row, (utts, tag) in enumerate([(train_utts, "train"), (test_utts, "held-out")]):
    y = np.asarray(utts[0], np.float64); yh = reconstruct(model, y, CFG)
    for col, (sig, name) in enumerate([(y, "target"), (yh, "prediction")]):
        ax[row, col].specgram(sig, NFFT=1024, Fs=CFG.fs_hi, noverlap=768, vmin=-140, vmax=-40)
        ax[row, col].axhline(CFG.fs_lo / 2, color="crimson", ls="--", lw=1)
        ax[row, col].set_title(f"{tag} {name}   peak {np.abs(sig).max():.3f}  rms {np.sqrt(np.mean(sig**2)):.4f}")
plt.tight_layout(); plt.show()
