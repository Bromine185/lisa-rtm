# ============================================================ OV3-8 launch on the measured-best configuration
# Measured on an A100-80GB after e7 set the training flags (deterministic OFF, cudnn.benchmark ON, TF32 ON):
#   sequential batch 32          589 ms/step  13.2 GB  54.3 audio-s per compute-s  (== the 8 Sep 587 ms)
#   stacked GROUP_MAX=1 batch 32 467 ms/step   7.0 GB  68.5
#   stacked GROUP_MAX=1 batch 64 925 ms/step  13.4 GB  69.2   <- throughput is flat in batch
# Grouping arms (GROUP_MAX 2 or 7) is slower at 2-4x the memory, and CUDA streams do nothing or hurt.
# Everything here writes to ROOT (the Drive mount), never to the Colab disk: checkpoints, the history JSON
# and the curves figure are rewritten every CKPT_EVERY steps so an interrupted session is still evaluable.
import gc, sys, json, torch

FAST["GROUP_MAX"], FAST["STREAMS"] = 1, False
BATCH = int(globals().get("BATCH_OVERRIDE", 64))
SEG = int(globals().get("SEG_OVERRIDE", 48000))
OV3_TAG = globals().get("OV3_TAG", "OV3_fast")
BUDGET_H = float(globals().get("BUDGET_H", 4.0))
CKPT_EVERY = int(globals().get("CKPT_EVERY", 500))

for _n in ("models_ov3", "hist_ov3"):
    if _n in globals():
        del globals()[_n]
sys.last_traceback = sys.last_value = sys.last_type = None
gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

assert isinstance(corpus, HostCorpus), "build the HostCorpus first"
steps_per_epoch = corpus.hours * 3600 / (BATCH * SEG / CFG.fs_hi)
dt = time_ov3_fast(corpus, ARMS, BATCH)
steps = max(2000, (int(BUDGET_H * 3600 / dt) // 1000) * 1000)
if isinstance(globals().get("STEPS_OVERRIDE"), int):
    steps = int(STEPS_OVERRIDE)
plan = ("plan: %d arms x %d steps = %.1f epochs of %.1f h; est %.2f h at %.0f ms/step; %.0f steps/epoch; "
        "batch %d x %.1f s; val %d utts; save every %d steps; flags %s"
        % (len(ARMS), steps, steps / steps_per_epoch, corpus.hours, steps * dt / 3600, dt * 1000,
           steps_per_epoch, BATCH, SEG / CFG.fs_hi, val_corpus.n, CKPT_EVERY,
           {k: FAST[k] for k in ("SUBPIXEL", "AMP", "COMPILE", "GROUP_MAX", "JITTER_PER_CELL")}))
print(plan, flush=True)
(ROOT / ("train_%s.log" % OV3_TAG)).write_text(plan + chr(10) + json.dumps(ARMS) + chr(10))

on_step, on_epoch = make_dashboard(OV3_TAG, ARMS)
models_ov3, hist_ov3 = train_ov3_fast(corpus, val_corpus, ARMS, steps, BATCH, 1e-3,
                                      (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3, CKPT_EVERY, OV3_TAG,
                                      test_utts[0], on_step=on_step, on_epoch=on_epoch,
                                      steps_per_epoch=steps_per_epoch)
json.dump({k: hist_ov3[k] for k in ARMS}, open(ROOT / ("ov3_history_%s.json" % OV3_TAG), "w"))
print("TRAINING DONE", OV3_TAG, flush=True)
