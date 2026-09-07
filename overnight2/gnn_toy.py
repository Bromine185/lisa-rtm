"""Graph toy: over-smoothing has two causes and the ensemble-mean test separates them.

Target on a graph = smooth predictable part f(x) + an UNPREDICTABLE high-graph-frequency part whose
amplitude depends on x.  Four models, one seed:
    shallow GNN (with self/root weight) x {MSE, energy score}
    deep GNN (8 pure propagation layers, no skip)   x {MSE, energy score}
Measured in the graph Fourier basis: energy ratio prediction/target per graph-frequency band, for a
single draw and for the ensemble mean, plus CRPS.

Predictions:
  * MSE, shallow: high-band deficit (the conditional mean has no unpredictable energy) = statistical.
  * ES,  shallow: single draw ~0 dB deficit; ensemble mean shows the SAME deficit as MSE.
  * ES,  deep:    single draw STILL shows a deficit = architectural (propagation is a low-pass filter).
So: deficit of the samples = what the architecture cannot express; deficit of the mean minus that =
what the objective (conditional expectation) removes.  Audio case: samples fix it -> objective.
"""
import json, math, sys, time, pathlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
OUT = pathlib.Path(__file__).resolve().parent / "gnn_toy_results.json"

# ---- graph: random geometric graph on the unit square ------------------------------------------
N = 600
pos = np.random.rand(N, 2)
D2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
np.fill_diagonal(D2, np.inf)
knn = np.argsort(D2, 1)[:, :8]                  # symmetric 8-NN graph: connected, no isolated nodes
A = np.zeros((N, N)); A[np.arange(N)[:, None], knn] = 1; A = np.maximum(A, A.T)
deg = A.sum(1)
L = np.diag(deg) - A
lam, U = np.linalg.eigh(L)                      # graph Fourier basis, lam ascending
A_hat = np.diag(1 / np.sqrt(deg + 1)) @ (A + np.eye(N)) @ np.diag(1 / np.sqrt(deg + 1))
A_hat_t = torch.tensor(A_hat, dtype=torch.float32)
U_t = torch.tensor(U, dtype=torch.float32)

K_LO, K_HI = 40, 200                            # input lives in the lowest 40 modes; noise in modes >= 200
D_IN = 4

def make_batch(B):
    '''x: smooth graph signals (B, N, D_IN); y: f(x) + heteroscedastic high-graph-frequency noise (B, N).'''
    c = np.random.randn(B, K_LO, D_IN) / np.sqrt(K_LO)
    x = np.einsum("nk,bkd->bnd", U[:, :K_LO], c) * np.sqrt(N)
    f = np.tanh(x[..., 0] * 2) + 0.5 * x[..., 1] * x[..., 2] - 0.3 * x[..., 3] ** 2
    sigma = 0.6 * (1 + np.tanh(x[..., 0]))       # amplitude depends on the input -> conditional structure
    z = np.random.randn(B, N - K_HI)
    eta = np.einsum("nk,bk->bn", U[:, K_HI:], z) * np.sqrt(N / (N - K_HI))
    y = f + sigma * eta
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class GNN(nn.Module):
    def __init__(self, depth, skip, n_noise, hidden=64):
        super().__init__()
        self.depth, self.skip, self.n_noise = depth, skip, n_noise
        d = D_IN + n_noise
        self.inp = nn.Linear(d, hidden)
        self.nb = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)])
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(depth)]) if skip else None
        self.out = nn.Linear(hidden, 1)
    def forward(self, x, eps=None):
        if self.n_noise:
            if eps is None:
                eps = torch.zeros(*x.shape[:2], self.n_noise)
            x = torch.cat([x, eps], -1)
        h = F.relu(self.inp(x))
        for i in range(self.depth):
            m = torch.einsum("nm,bmh->bnh", A_hat_t, self.nb[i](h))
            h = F.relu(m + self.self_w[i](h)) if self.skip else F.relu(m)
        return self.out(h).squeeze(-1)

def train(model, kind, steps=1500, B=16, lr=2e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for s in range(steps):
        x, y = make_batch(B)
        if kind == "mse":
            loss = F.mse_loss(model(x), y)
        else:                                                     # energy score, 2 draws, Euclidean norm over nodes
            e1, e2 = torch.randn(B, N, model.n_noise), torch.randn(B, N, model.n_noise)
            y1, y2 = model(x, e1), model(x, e2)
            d = lambda a, b: (a - b).norm(dim=1)
            loss = (0.5 * (d(y, y1) + d(y, y2)) - 0.5 * d(y1, y2)).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model

BANDS = [(0, 40), (40, 120), (120, 200), (200, 400), (400, 600)]
def band_ratio(pred, y):
    '''10 log10 of energy(pred)/energy(target) per graph-frequency band, pooled over the batch.'''
    P, Y = pred @ U_t, y @ U_t                                    # graph Fourier coefficients (B, N)
    return [10 * math.log10(float((P[:, a:b] ** 2).sum() / (Y[:, a:b] ** 2).sum())) for a, b in BANDS]

def crps(samples, truth):
    M = samples.shape[0]
    t1 = (samples - truth[None]).abs().mean(0)
    t2 = (samples[:, None] - samples[None]).abs().mean((0, 1))
    return float((t1 - 0.5 * t2).mean())

@torch.no_grad()
def evaluate(model, kind, M=32):
    torch.manual_seed(123); np.random.seed(123)
    x, y = make_batch(64)
    if kind == "mse":
        p = model(x)
        return {"single": band_ratio(p, y), "mean": band_ratio(p, y), "crps": crps(p[None], y), "rmse_mean": float(((p - y) ** 2).mean().sqrt())}
    S = torch.stack([model(x, torch.randn(64, N, model.n_noise)) for _ in range(M)])
    return {"single": band_ratio(S[0], y), "mean": band_ratio(S.mean(0), y), "crps": crps(S, y),
            "rmse_mean": float(((S.mean(0) - y) ** 2).mean().sqrt()), "rmse_single": float(((S[0] - y) ** 2).mean().sqrt())}

RES = {}
t0 = time.time()
for arch, (depth, skip) in {"shallow": (2, True), "deep": (8, False)}.items():
    for kind in ("mse", "es"):
        torch.manual_seed(0)
        m = train(GNN(depth, skip, n_noise=4), kind)
        r = evaluate(m, kind)
        RES[f"{arch}/{kind}"] = r
        print(f"{arch:8s} {kind:4s}  single-draw deficit by band {['%+.1f' % v for v in r['single']]}  "
              f"mean-of-32 {['%+.1f' % v for v in r['mean']]}  CRPS {r['crps']:.4f}  rmse(mean) {r['rmse_mean']:.4f}  [{time.time()-t0:.0f}s]", flush=True)
RES["bands"] = BANDS
json.dump(RES, open(OUT, "w"), indent=1)
print("GNN TOY DONE")
