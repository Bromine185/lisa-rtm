# ============================================================ XL-4 timing + launch
# Four arms, one data stream, identical batches:
#   relu_l1e-3  faithful LISA (official lambda)          -- the H1' measurement
#   relu_l1e-2, relu_l1e-1  the lambda sweep              -- what does the spectral term buy, and at what cost
#   ff_l1e-3    Fourier-feature decoder, same lambda      -- is over-smoothing spectral bias or the loss?
ARMS = {"relu_l1e-3": ("relu", 1e-3), "relu_l1e-2": ("relu", 1e-2),
        "relu_l1e-1": ("relu", 1e-1), "ff_l1e-3": ("ff", 1e-3)}
BATCH, SEG, BUDGET_H = 64, 48000, 3.5
XL_TAG = "XL4_b64_1s"

corpus = GPUCorpus(train_utts, CFG, seg_hi=SEG)
dt1 = time_steps(corpus, 1, BATCH)
dt4 = time_steps(corpus, len(ARMS), BATCH)
steps_50ep = int(round(50 * corpus.hours * 3600 / BATCH))
steps = min(steps_50ep, int(BUDGET_H * 3600 / dt4))
steps = max(2000, (steps // 1000) * 1000)
epochs = steps * BATCH / (corpus.hours * 3600)
plan = (f"plan: {len(ARMS)} arms x {steps} steps = {epochs:.1f} epochs of {corpus.hours:.1f} h "
        f"(50-epoch target {steps_50ep}); est {steps*dt4/3600:.2f} h at {dt4*1000:.0f} ms/step")
print(plan)
(ROOT / f"train_{XL_TAG}.log").write_text(plan + chr(10) + json.dumps(ARMS) + chr(10))
probe = test_utts[0]
models_xl, hist_xl = train_arms(corpus, ARMS, steps, BATCH, 1e-3, (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3,
                                1000, XL_TAG, probe)
print("TRAINING DONE", XL_TAG)
