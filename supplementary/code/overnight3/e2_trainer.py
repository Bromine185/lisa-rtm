# ============================================================ OV3-2 paired trainer with validation loss and curves
# train_ov2 (one batch stream, one optimiser per arm) plus: a fixed held-out validation set on which every
# arm's OWN objective is scored every val_every steps (fixed noise seed, no anchor jitter: the decoder path
# reconstruct() uses), an EMA of the training loss at the same steps, the history JSON written at every checkpoint,
# and a curves figure (train/val loss, val terms, probe SNR + deficit) redrawn at every checkpoint.
import copy, math, time, json, numpy as np, torch, matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR


def val_batches(val_corpus, tag, n, batch):
    rng = stream(f"{tag}/val")
    return [val_corpus.batch(rng, batch) for _ in range(n)]


def _fork_devices():
    return [torch.cuda.current_device()] if (torch.cuda.is_available() and DEVICE.type == "cuda") else []


@torch.no_grad()
def val_loss(kind, lam, m, vb, spec_loss, seed=1234):
    '''Mean (loss, wave, spec) of the arm's own objective over the fixed validation batches, eps drawn from a
    fixed seed and perturb=False (no anchor jitter, as in reconstruct), so successive evaluations differ only
    through the weights.'''
    m.eval()
    tot = np.zeros(3)
    with torch.random.fork_rng(devices=_fork_devices()):
        torch.manual_seed(seed)
        for x, y in vb:
            loss, t = arm_loss3(kind, lam, m, x, y, spec_loss, perturb=False)
            tot += (loss.item(), t["wave"], t["spec"])
    m.train()
    return tot / len(vb)


def _nan(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def _f(vs):
    return [float("nan") if _nan(v) else v for v in vs]


def plot_curves(hist, tag, path=None):
    names = list(hist)
    cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for i, k in enumerate(names):
        h, c = hist[k], cols[i % len(cols)]
        if h["val_step"]:
            ax[0].plot(h["val_step"], h["train_loss_ema"], "-", color=c, lw=1.2, label=f"{k} train (EMA)")
            ax[0].plot(h["val_step"], h["val_loss"], "--", color=c, lw=1.2, label=f"{k} val")
            ax[1].plot(h["val_step"], h["val_wave"], "-", color=c, lw=1.2, label=f"{k} wave")
            ax[1].plot(h["val_step"], h["val_spec"], ":", color=c, lw=1.2, label=f"{k} spec")
        if h["dev_step"]:
            ax[2].plot(h["dev_step"], h["snr0"], "-", color=c, lw=1.2, label=f"{k} SNR tau=0")
            if not all(_nan(v) for v in h["snr1"]):
                ax[2].plot(h["dev_step"], _f(h["snr1"]), "-.", color=c, lw=1.0, label=f"{k} SNR tau=1")
    ax[0].set_yscale("log"); ax[0].set_title("objective: train EMA (solid) vs held-out val (dashed)", fontsize=9)
    ax[0].set_xlabel("step"); ax[0].legend(fontsize=5, ncol=2)
    ax[1].set_yscale("log"); ax[1].set_title("val terms: wave (solid), spec (dotted)", fontsize=9); ax[1].set_xlabel("step"); ax[1].legend(fontsize=5, ncol=2)
    ax[2].set_title("probe utterance: SNR (left), HB deficit dB (right, dashed)", fontsize=9); ax[2].set_xlabel("step"); ax[2].set_ylabel("SNR dB")
    if any(hist[k]["snr_naive"] for k in names):
        ax[2].axhline(hist[names[0]]["snr_naive"][0], color="grey", ls=":", lw=1)
    ax2 = ax[2].twinx()
    for i, k in enumerate(names):
        h, c = hist[k], cols[i % len(cols)]
        if h["dev_step"]:
            d = h["def1"] if not all(_nan(v) for v in h["def1"]) else h["def0"]
            ax2.plot(h["dev_step"], _f(d), "--", color=c, lw=1.0)
    ax2.set_ylabel("deficit dB (one draw for samplers)"); ax2.axhline(0, color="k", lw=0.6)
    ax[2].legend(fontsize=5, ncol=2)
    plt.tight_layout()
    path = path or (FIGS / f"ov3_curves_{tag}.png")
    plt.savefig(path, dpi=130); plt.close(fig)
    return path


def train_ov3(corpus, val_corpus, arms, steps, batch, lr, milestones, gamma, clip, ckpt_every, tag, probe,
              val_every=500, n_val_batches=8, log_every=25):
    '''arms: {name: (kind, lam[, cls_name])}.  One batch stream; one init per class, the LISASD init sharing
    every LISAS weight (copy_shared) so the two classes start identical at eps=0; one optimiser per arm.'''
    names = list(arms)
    specs = {k: arm3(arms[k]) for k in names}
    base = {"LISAS": LISAS(CFG).to(DEVICE)}
    if any(specs[k][2] == "LISASD" for k in names):
        base["LISASD"] = copy_shared(base["LISAS"], LISASD(CFG).to(DEVICE))
    models = {k: copy.deepcopy(base[specs[k][2]]) for k in names}
    for k in names:
        models[k].tau = 0.0 if specs[k][0].startswith("det") else 1.0
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=lr) for k in names}
    scheds = {k: MultiStepLR(opts[k], [int(f * steps) for f in milestones], gamma) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    hist = {k: {"step": [], "wave": [], "spec": [], "spread": [], "lr": [],
                "val_step": [], "val_loss": [], "val_wave": [], "val_spec": [], "train_loss_ema": [],
                "dev_step": [], "snr0": [], "def0": [], "snr1": [], "def1": [], "snr_naive": []} for k in names}
    ema = {k: None for k in names}
    vb = val_batches(val_corpus, tag, n_val_batches, batch)
    rng = stream(f"{tag}/batches")
    run_dir = CKPT / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    probe = np.asarray(probe, np.float64)
    naive = snr_db(probe, naive_upsample(probe, CFG))
    logf = open(ROOT / f"train_{tag}.log", "a")
    hist_path = ROOT / f"ov3_history_{tag}.json"
    t0 = time.time()
    for step in range(steps):
        x, y = corpus.batch(rng, batch)
        log = (step % log_every == 0)
        for k in names:
            kind, lam, _ = specs[k]
            m = models[k]
            m.train()
            loss, terms = arm_loss3(kind, lam, m, x, y, spec_loss)
            opts[k].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            opts[k].step()
            scheds[k].step()
            lv = loss.item()
            ema[k] = lv if ema[k] is None else 0.98 * ema[k] + 0.02 * lv
            if log:
                h = hist[k]
                h["step"].append(step); h["wave"].append(terms["wave"]); h["spec"].append(terms["spec"])
                h["spread"].append(terms.get("spread", 0.0)); h["lr"].append(scheds[k].get_last_lr()[0])
        if (step + 1) % val_every == 0 or step + 1 == steps:
            vline = f"  val {step+1:>6}"
            for k in names:
                kind, lam, _ = specs[k]
                vl, vw, vs = val_loss(kind, lam, models[k], vb, spec_loss)
                h = hist[k]
                h["val_step"].append(step + 1); h["val_loss"].append(float(vl)); h["val_wave"].append(float(vw))
                h["val_spec"].append(float(vs)); h["train_loss_ema"].append(float(ema[k]))
                vline += f" | {k}: train {ema[k]:.4f} val {vl:.4f}"
            print(vline, flush=True); logf.write(vline + chr(10)); logf.flush()
        if (step + 1) % ckpt_every == 0 or step + 1 == steps:
            line = f"step {step+1:>6}/{steps} [{time.time()-t0:.0f}s]"
            for k in names:
                kind, lam, cls_name = specs[k]
                m = models[k]
                pm = probe_metrics(m, probe, CFG, naive)
                h = hist[k]
                h["dev_step"].append(step + 1); h["snr_naive"].append(naive)
                h["snr0"].append(pm[0.0][0]); h["def0"].append(pm[0.0][1])
                s1, d1 = pm.get(1.0, (None, None))                   # None (not NaN): valid JSON for det arms
                h["snr1"].append(s1); h["def1"].append(d1)
                save_ckpt({"model": m.state_dict(), "step": step + 1, "history": h, "arm": (kind, lam, cls_name),
                           "n_noise": m.n_noise, "n_dec": getattr(m, "n_dec", 0), "cls": cls_name,
                           "batch": batch, "seg": corpus.seg_hi, "tag": tag}, run_dir / f"{k}.pt")
                line += f" | {k}: w {h['wave'][-1]:.4f} s {h['spec'][-1]:.3f} SNR0 {pm[0.0][0]:5.2f} def0 {pm[0.0][1]:+6.2f}"
                if 1.0 in pm:
                    line += f" SNR1 {s1:5.2f} def1 {d1:+6.2f}"
            line += f"  (naive {naive:.2f})"
            print(line, flush=True)
            logf.write(line + chr(10)); logf.flush()
            try:                                                       # Drive mount can flake; ckpts already saved
                json.dump({k: hist[k] for k in names}, open(hist_path, "w"))
            except OSError as e:
                print("history write failed:", repr(e), flush=True)
            try:
                plot_curves(hist, tag)
            except Exception as e:
                print("plot_curves failed:", repr(e), flush=True)
    logf.close()
    return models, hist


def time_ov3(corpus, arms, batch, n=20):
    names = list(arms)
    specs = {k: arm3(arms[k]) for k in names}
    models = {k: CLASSES[specs[k][2]](CFG).to(DEVICE) for k in names}
    opts = {k: torch.optim.Adam(models[k].parameters(), lr=1e-3) for k in names}
    spec_loss = MultiScaleSTFTLoss(CFG.n_fft).to(DEVICE)
    rng = stream("timing")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for i in range(n + 3):
        if i == 3:
            sync(); t0 = time.time()
        x, y = corpus.batch(rng, batch)
        for k in names:
            loss, _ = arm_loss3(specs[k][0], specs[k][1], models[k], x, y, spec_loss)
            opts[k].zero_grad(set_to_none=True); loss.backward(); opts[k].step()
    sync()
    dt = (time.time() - t0) / n
    mem = f", peak mem {torch.cuda.max_memory_allocated()/1e9:.1f} GB" if torch.cuda.is_available() else ""
    print(f"{len(names)} arm(s) {names}, batch {batch} x {corpus.seg_hi}: {dt*1000:.0f} ms/step{mem}", flush=True)
    del models, opts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dt

print("OV3 trainer defined (train_ov3, time_ov3, val_loss, plot_curves).", flush=True)
