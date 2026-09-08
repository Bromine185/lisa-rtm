# ============================================================ OV2-8 deterministic lambda sweep for LSD + ViSQOL
# On LSD and ViSQOL (both modes) the 4 Sep lambda = 1e-1 model beat every other condition, and the trend
# 1e-3 -> 1e-2 -> 1e-1 was monotone.  This extends the sweep upward: paired arms on identical batches,
# lambda in {0.1, 0.3, 1.0}, the paper's loss otherwise.  Evaluate with baseband passthrough as well
# (c7), which removes the phase-blind term's damage to the baseband that killed the lambda = 1.0 run on
# 1 Sep (SNR -5.8 dB) -- with passthrough the model is only ever responsible for the band above 6 kHz.
# Requires: boot (full corpus, c0_boot -- needs >= 40 GB host RAM, i.e. an A100 runtime), c1_model.
ARMS_L = {"det_l0.1": ("det", 0.1), "det_l0.3": ("det", 0.3), "det_l1.0": ("det", 1.0)}
BATCH, SEG = 32, 48000
LAM_TAG = "OV2_lambda"
BUDGET_L = globals().get("BUDGET_L", 3.0)
import gc, sys
sys.last_traceback = sys.last_value = sys.last_type = None
gc.collect(); torch.cuda.empty_cache()
if "corpus" not in globals():
    corpus = HostCorpus(train_utts, CFG, seg_hi=SEG)
    _step = max(1, len(train_utts) // 200)
    train_utts, train_spk = train_utts[::_step][:200], train_spk[::_step][:200]
    gc.collect()
MODEL_CLS = LISAS
dt = time_ov2(corpus, ARMS_L, BATCH)
steps = max(2000, (int(BUDGET_L * 3600 / dt) // 1000) * 1000)
plan = f"plan: {len(ARMS_L)} lambda arms x {steps} steps = {steps*BATCH/(corpus.hours*3600):.1f} epochs; est {steps*dt/3600:.2f} h at {dt*1000:.0f} ms/step"
print(plan, flush=True)
(ROOT / f"train_{LAM_TAG}.log").write_text(plan + chr(10) + json.dumps(ARMS_L) + chr(10))
models_lam, hist_lam = train_ov2(corpus, ARMS_L, steps, BATCH, 1e-3, (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3,
                                 1000, LAM_TAG, test_utts[0])
json.dump({k: hist_lam[k] for k in ARMS_L}, open(ROOT / f"ov2_history_{LAM_TAG}.json", "w"))
print("LAMBDA TRAINING DONE", flush=True)
