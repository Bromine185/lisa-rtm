# ============================================================ OV2-2 timing + launch
ARMS = {
    "det":       ("det",       1e-2),   # the paper's loss at the lambda the frontier run found best
    "det_split": ("det_split", 1e-2),   # deterministic band-split: no waveform term above 6 kHz
    "es_marg":   ("es_marg",   1e-2),   # energy score, per-sample / per-bin marginals
    "es_slice":  ("es_slice",  1e-2),   # energy score, sliced over the frame -> proper for the joint law
    "es_wave":   ("es_wave",   0.0),    # energy score on the waveform alone
}
BATCH, SEG = 32, 48000                 # A100-40GB: two-draw arms run 2*BATCH per forward
OV2_TAG = "OV2_es"
BUDGET_H = globals().get("BUDGET_H", 4.3)

# free whatever a failed cell left on the GPU (corpus tensors, traceback frames)
import gc, sys
for _n in ("corpus", "models_ov2", "hist_ov2"):
    if _n in globals():
        del globals()[_n]
sys.last_traceback = sys.last_value = sys.last_type = None
gc.collect(); torch.cuda.empty_cache()
print(f"GPU before corpus: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved", flush=True)

corpus = HostCorpus(train_utts, CFG, seg_hi=SEG)
# keep only the 200-utterance fit subset the ladder needs; the full list is 26 GB of RAM
_step = max(1, len(train_utts) // 200)
train_utts, train_spk = train_utts[::_step][:200], train_spk[::_step][:200]
gc.collect()
dt = time_ov2(corpus, ARMS, BATCH)
steps = int(BUDGET_H * 3600 / dt)
steps = max(2000, (steps // 1000) * 1000)
epochs = steps * BATCH / (corpus.hours * 3600)
plan = (f"plan: {len(ARMS)} arms x {steps} steps = {epochs:.1f} epochs of {corpus.hours:.1f} h; "
        f"est {steps*dt/3600:.2f} h at {dt*1000:.0f} ms/step")
print(plan, flush=True)
(ROOT / f"train_{OV2_TAG}.log").write_text(plan + chr(10) + json.dumps(ARMS) + chr(10))
probe = test_utts[0]
models_ov2, hist_ov2 = train_ov2(corpus, ARMS, steps, BATCH, 1e-3, (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3,
                                 1000, OV2_TAG, probe)
json.dump({k: hist_ov2[k] for k in ARMS}, open(ROOT / f"ov2_history_{OV2_TAG}.json", "w"))
print("TRAINING DONE", OV2_TAG, flush=True)
