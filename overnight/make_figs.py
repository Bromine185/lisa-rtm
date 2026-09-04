"""Pareto figure: what the spectral weight buys and what it costs."""
import json, pathlib, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = pathlib.Path(__file__).resolve().parents[1]
R = json.load(open(REPO / "overnight/local_eval_XL4.json"))
arms = [("relu_l1e-3", 1e-3), ("relu_l1e-2", 1e-2), ("relu_l1e-1", 1e-1)]
lam = [l for _, l in arms]
get = lambda k, f: [R[a][k][f] for a, _ in arms]
snr, naive, defi, lsd = get("spread36","snr"), get("spread36","snr_naive"), get("spread36","deficit"), get("spread36","lsd")

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
ax[0].semilogx(lam, defi, "o-", color="#c0392b")
ax[0].axhspan(-2, 2, color="grey", alpha=0.15)
ax[0].set_xlabel(r"$\lambda_{spec}$"); ax[0].set_ylabel("high-band deficit (dB)")
ax[0].set_title("What the spectral term buys")
for x, y in zip(lam, defi): ax[0].annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

ax[1].semilogx(lam, np.array(snr) - np.array(naive), "o-", color="#2c3e50")
ax[1].axhline(0, color="grey", ls="--"); ax[1].set_xlabel(r"$\lambda_{spec}$")
ax[1].set_ylabel("SNR $-$ naive upsample (dB)"); ax[1].set_title("What it costs")
for x, y in zip(lam, np.array(snr) - np.array(naive)): ax[1].annotate(f"{y:+.2f}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)

ax[2].plot(np.array(snr) - np.array(naive), lsd, "o-", color="#8e44ad")
for (a, l), x, y in zip(arms, np.array(snr) - np.array(naive), lsd):
    ax[2].annotate(rf"$\lambda$={l:g}", (x, y), textcoords="offset points", xytext=(6, -3), fontsize=8)
ax[2].axhline(0.81, color="green", ls=":", label="paper LSD 0.81")
ax[2].set_xlabel("SNR $-$ naive (dB)"); ax[2].set_ylabel("LSD"); ax[2].set_title("The frontier"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(REPO / "overnight/pareto_lambda.png", dpi=140)
print("wrote pareto_lambda.png")

# architecture control
fig, ax = plt.subplots(figsize=(5.2, 3.8))
pairs = [("relu_l1e-3", "ReLU decoder"), ("ff_l1e-3", "Fourier-feature decoder")]
vals = [[R[a]["spread36"][f] for a, _ in pairs] for f in ("deficit", "snr", "lsd")]
x = np.arange(2)
ax.bar(x - 0.2, vals[0], 0.4, label="deficit (dB)", color="#c0392b")
ax.bar(x + 0.2, [v - 15 for v in vals[1]], 0.4, label="SNR $-$ 15 (dB)", color="#2c3e50")
ax.set_xticks(x); ax.set_xticklabels([n for _, n in pairs], fontsize=8)
ax.set_title(r"Architecture control at $\lambda=10^{-3}$"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(REPO / "overnight/arch_control.png", dpi=140)
print("wrote arch_control.png")
print(json.dumps({a: {k: round(R[a]["spread36"][k], 3) for k in ("snr","snr_naive","lsd","hb_lsd","deficit","baseband")} for a in R}, indent=1))
