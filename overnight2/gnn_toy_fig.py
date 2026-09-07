import json, pathlib, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent
R = json.load(open(HERE / "gnn_toy_results.json"))
bands = R["bands"]
x = np.arange(len(bands))
labels = [f"{a}-{b}" for a, b in bands]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for a, arch in zip(ax, ("shallow", "deep")):
    a.plot(x, R[f"{arch}/mse"]["single"], "s-", color="k", label="MSE (point predictor)")
    a.plot(x, R[f"{arch}/es"]["single"], "o-", color="crimson", label="energy score, one draw")
    a.plot(x, R[f"{arch}/es"]["mean"], "o--", color="crimson", alpha=0.6, label="energy score, mean of 32 draws")
    a.axhline(0, color="grey", lw=1); a.axhspan(-2, 2, color="grey", alpha=0.12)
    a.set_xticks(x); a.set_xticklabels(labels, fontsize=8)
    a.set_xlabel("graph-frequency band (Laplacian mode index)")
    a.set_title({"shallow": "2-layer GNN with root weight", "deep": "8 propagation layers, no skip"}[arch], fontsize=10)
    a.grid(alpha=0.3)
ax[0].set_ylabel("pred / target energy (dB)")
ax[0].legend(fontsize=8, loc="lower left")
ax[0].annotate("objective-induced:\nsamples fix it", (3, -8), fontsize=8, color="crimson", ha="center")
ax[1].annotate("architecture-induced:\nsamples cannot", (3, -8), fontsize=8, color="crimson", ha="center")
fig.suptitle("Two causes of over-smoothing on a graph, separated by the ensemble-mean test", fontsize=11)
plt.tight_layout(); plt.savefig(HERE / "gnn_toy.png", dpi=140)
print("saved", HERE / "gnn_toy.png")
