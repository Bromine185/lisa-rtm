"""Graph toy, part 2: the geometry of the scoring rule decides which structure the sampler learns.

Same graph, data and shallow GNN as gnn_toy.py.  Three strictly-proper-for-something objectives:
    crps_node   sum over nodes of univariate CRPS        -> proper for each node's MARGINAL only
    es_eucl     energy score, Euclidean norm over nodes  -> proper for the JOINT law of all nodes
    es_sliced   energy score, |theta . (y - y')| averaged over 64 random unit directions in R^N
                                                        -> proper for the joint law (Cramer-Wold), 1-D power
Measured: per-node CRPS (what crps_node optimises), CRPS of the graph-Fourier coefficients (joint
structure across nodes, a fixed orthogonal projection), and the single-draw energy ratio per band.

Prediction: crps_node gets the marginals right but puts its noise in the WRONG graph-frequency bands
(spatially white, so energy leaks into modes 40-200 where the target has none); the joint scores put
it where the target has it (modes >= 200).
"""
import json, math, time, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import importlib.util
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("toy", HERE / "gnn_toy.py")
src = (HERE / "gnn_toy.py").read_text().split("RES = {}")[0]        # reuse graph, data, model, metrics
G = {"__name__": "toy", "__file__": str(HERE / "gnn_toy.py")}
exec(compile(src, "gnn_toy.py", "exec"), G)
make_batch, GNN, band_ratio, crps, N, U_t = G["make_batch"], G["GNN"], G["band_ratio"], G["crps"], G["N"], G["U_t"]
torch.manual_seed(0); np.random.seed(0)

def train(model, kind, steps=1500, B=16, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        x, y = make_batch(B)
        e1, e2 = torch.randn(B, N, model.n_noise), torch.randn(B, N, model.n_noise)
        y1, y2 = model(x, e1), model(x, e2)
        if kind == "crps_node":
            d = lambda a, b: (a - b).abs().mean(dim=1)
        elif kind == "es_eucl":
            d = lambda a, b: (a - b).norm(dim=1)
        elif kind == "es_sliced":
            th = torch.randn(N, 64); th = th / th.norm(dim=0, keepdim=True)
            d = lambda a, b: ((a - b) @ th).abs().mean(dim=1)
        loss = (0.5 * (d(y, y1) + d(y, y2)) - 0.5 * d(y1, y2)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

@torch.no_grad()
def evaluate(model, M=32):
    torch.manual_seed(123); np.random.seed(123)
    x, y = make_batch(64)
    S = torch.stack([model(x, torch.randn(64, N, model.n_noise)) for _ in range(M)])
    return {"single": band_ratio(S[0], y), "mean": band_ratio(S.mean(0), y),
            "crps_node": crps(S, y), "crps_fourier": crps(S @ U_t, y @ U_t),
            "crps_fourier_hi": crps((S @ U_t)[..., 200:], (y @ U_t)[..., 200:]),
            "crps_fourier_mid": crps((S @ U_t)[..., 40:200], (y @ U_t)[..., 40:200])}

RES = {}
t0 = time.time()
for kind in ("crps_node", "es_eucl", "es_sliced"):
    torch.manual_seed(0)
    m = train(GNN(2, True, n_noise=4), kind)
    r = evaluate(m); RES[kind] = r
    print(f"{kind:10s} single-draw band ratio {['%+.1f' % v for v in r['single']]}  CRPS node {r['crps_node']:.4f} | "
          f"Fourier all {r['crps_fourier']:.4f} mid(40-200) {r['crps_fourier_mid']:.4f} hi(>=200) {r['crps_fourier_hi']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
json.dump(RES, open(HERE / "gnn_toy_scores.json", "w"), indent=1)
print("GNN SCORES DONE")
