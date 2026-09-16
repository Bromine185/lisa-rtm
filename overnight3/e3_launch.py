# ============================================================ OV3-3 timing + launch
# Seven paired arms on identical batches (16 Sep, after the literature synthesis): the det / es_marg references,
# the spectral weight raised x10 so the spectral geometry dominates the waveform term in the incoherent band,
# the waveform term restricted to the low band (es_split), the aggregate-proper ERB-band-energy term (es_erb),
# and the decoder-noise class LISASD on the last two.
# Requires in the kernel: c0 boot, c1_model, e1_model, e2_trainer.
ARMS = {
    "det":             ("det",               1e-2, "LISAS"),    # reference (8 Sep recipe)
    "es_marg":         ("es_marg",           1e-2, "LISAS"),    # reference sampler (8 Sep recipe)
    "es_marg_l0.1":    ("es_marg",           1e-1, "LISAS"),    # spectral term dominant (weight x10)
    "es_split_l0.1":   ("es_split_marg",     1e-1, "LISAS"),    # + waveform term on the low band only
    "es_erb_l0.1":     ("es_marg_erb",       1e-1, "LISAS"),    # + aggregate-proper term on log ERB-band energies
    "es_dec_l0.1":     ("es_marg",           1e-1, "LISASD"),   # noise at the output rate
    "es_dec_erb_l0.1": ("es_marg_erb",       1e-1, "LISASD"),   # both
}
if isinstance(globals().get("ARMS_OVERRIDE"), dict):
    ARMS = dict(ARMS_OVERRIDE)
    print("ARMS_OVERRIDE in effect:", list(ARMS), flush=True)
BATCH, SEG = 32, 48000                 # A100-40GB: two-draw arms run 2*BATCH per forward
OV3_TAG = "OV3_es"
BUDGET_H = globals().get("BUDGET_H", 4.3)

# free whatever a failed cell left on the GPU (corpus tensors, traceback frames)
import gc, sys, json
for _n in ("models_ov2", "hist_ov2", "models_ov3", "hist_ov3"):
    if _n in globals():
        del globals()[_n]
sys.last_traceback = sys.last_value = sys.last_type = None
gc.collect(); torch.cuda.empty_cache()
print(f"GPU before corpus: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated, {torch.cuda.memory_reserved()/1e9:.2f} GB reserved", flush=True)

# The Hub loader leaves a dict named `corpus`, so test the TYPE, not the name.
if not isinstance(globals().get("corpus"), HostCorpus):
    corpus = HostCorpus(train_utts, CFG, seg_hi=SEG)
    val_corpus = HostCorpus(test_utts[12:52], CFG, seg_hi=SEG)  # test speakers (28 of p236 + 12 of p237), utterances disjoint from EVAL12
    # keep only the 200-utterance fit subset the ladder needs; the full list is 26 GB of RAM
    _step = max(1, len(train_utts) // 200)
    train_utts, train_spk = train_utts[::_step][:200], train_spk[::_step][:200]
    gc.collect()
else:
    print("reusing corpus / val_corpus from the kernel", flush=True)
dt = time_ov3(corpus, ARMS, BATCH)
steps = int(BUDGET_H * 3600 / dt)
steps = max(2000, (steps // 1000) * 1000)
if isinstance(globals().get("STEPS_OVERRIDE"), int):
    steps = int(STEPS_OVERRIDE)
    print("STEPS_OVERRIDE in effect:", steps, flush=True)
epochs = steps * BATCH / (corpus.hours * 3600)
plan = (f"plan: {len(ARMS)} arms x {steps} steps = {epochs:.1f} epochs of {corpus.hours:.1f} h; "
        f"est {steps*dt/3600:.2f} h at {dt*1000:.0f} ms/step; val {val_corpus.n} utts (test_utts[12:52]: test speakers, disjoint from EVAL12 in utterance, not speaker)")
print(plan, flush=True)
(ROOT / f"train_{OV3_TAG}.log").write_text(plan + chr(10) + json.dumps(ARMS) + chr(10))
probe = test_utts[0]
models_ov3, hist_ov3 = train_ov3(corpus, val_corpus, ARMS, steps, BATCH, 1e-3, (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3,
                                 1000, OV3_TAG, probe)
json.dump({k: hist_ov3[k] for k in ARMS}, open(ROOT / f"ov3_history_{OV3_TAG}.json", "w"))
print("TRAINING DONE", OV3_TAG, flush=True)
