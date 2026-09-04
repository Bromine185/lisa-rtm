# ============================================================ XL-5 evaluate every arm
import matplotlib.pyplot as plt
ARM_ORDER = list(models_xl)
spread = lambda xs, n: xs[:: max(1, len(xs) // n)][:n]
train_eval = spread(train_utts, 12)
paper_eval = spread(paper_test_utts, 100)
RES = {}
for k in ARM_ORDER:
    m = models_xl[k]; m.eval()
    RES[k] = {"ours12": evaluate_model(m, test_utts, 12, f"{k} ours(12)"),
              "ours120": evaluate_model(m, test_utts, 120, f"{k} ours(120)"),
              "paper12": evaluate_model(m, paper_eval, 12, f"{k} paper(12)"),
              "paper100": evaluate_model(m, paper_eval, 100, f"{k} paper(100)"),
              "train12": evaluate_model(m, train_eval, 12, f"{k} train(12)"),
              "phase": [float(v) for v in phase_swap(m, test_utts, 12, f"{k} phase-swap(12)")]}
    print()
json.dump(RES, open(ROOT / f"xl_results_{XL_TAG}.json", "w"), indent=1)

fig, ax = plt.subplots(figsize=(8, 4))
CURVES = {}
for k in ARM_ORDER:
    c, b = deficit_curve(models_xl[k], test_utts, 12)
    CURVES[k] = [c.tolist(), b.tolist()]
    ax.semilogx(c, b, "o-", ms=3, label=k)
ax.axhline(0, color="k", lw=1); ax.axvline(CFG.fs_lo / 2, color="crimson", ls="--"); ax.axhspan(-2, 2, color="grey", alpha=0.15)
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("pred/target energy (dB)")
ax.set_title(f"Energy ratio by arm, held-out p236-238 (12 utts), {corpus.hours:.0f} h train"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(FIGS / f"deficit_arms_{XL_TAG}.png", dpi=130); plt.show()
json.dump(CURVES, open(ROOT / f"xl_curves_{XL_TAG}.json", "w"))

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for k in ARM_ORDER:
    h = hist_xl[k]
    ax[0].plot(h["step"], h["wave"], lw=0.7, label=k)
    ax[1].plot(h["dev_step"], h["deficit"], "o-", ms=3, label=k)
    ax[2].plot(h["dev_step"], h["snr"], "o-", ms=3, label=k)
ax[0].set_yscale("log"); ax[0].set_title("waveform L1 (per logged batch)")
ax[1].set_title("high-band deficit on probe (dB)"); ax[1].axhline(0, color="k", lw=1)
ax[2].set_title("SNR on probe (dB)"); ax[2].axhline(hist_xl[ARM_ORDER[0]]["snr_naive"][-1], color="grey", ls="--", label="naive")
for a in ax:
    a.set_xlabel("step"); a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(FIGS / f"training_arms_{XL_TAG}.png", dpi=130); plt.show()
json.dump({k: hist_xl[k] for k in ARM_ORDER}, open(ROOT / f"xl_history_{XL_TAG}.json", "w"))
print("EVAL DONE")
