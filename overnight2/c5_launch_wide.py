# ============================================================ OV2-5 wide-context run (receptive-field test)
# Two arms, same batches, same init: the paper's loss and the marginal energy score, both on LISASW.
# The question is kappa: does 22 ms of context make any of the high band coherently predictable?
exec(open(OV2 + 'c1_model.py').read())        # picks up MODEL_CLS / cls-aware load_arm
exec(open(OV2 + 'c1b_wide.py').read())
MODEL_CLS = LISASW
ARMS_W = {"wide_det": ("det", 1e-2), "wide_es_marg": ("es_marg", 1e-2)}
BATCH, SEG = 32, 48000
WIDE_TAG = "OV2_wide"
BUDGET_W = globals().get("BUDGET_W", 1.3)
import gc, sys
sys.last_traceback = sys.last_value = sys.last_type = None
gc.collect(); torch.cuda.empty_cache()
if "corpus" not in globals():
    corpus = HostCorpus(train_utts_all if "train_utts_all" in globals() else train_utts, CFG, seg_hi=SEG)
dt = time_ov2(corpus, ARMS_W, BATCH)
steps = max(2000, (int(BUDGET_W * 3600 / dt) // 1000) * 1000)
plan = f"plan: {len(ARMS_W)} wide arms x {steps} steps = {steps*BATCH/(corpus.hours*3600):.1f} epochs; est {steps*dt/3600:.2f} h at {dt*1000:.0f} ms/step"
print(plan, flush=True)
(ROOT / f"train_{WIDE_TAG}.log").write_text(plan + chr(10) + json.dumps(ARMS_W) + chr(10))
models_wide, hist_wide = train_ov2(corpus, ARMS_W, steps, BATCH, 1e-3, (0.2, 0.4, 0.5, 0.6, 0.7, 0.8), 0.5, 1e-3,
                                   1000, WIDE_TAG, test_utts[0])
json.dump({k: hist_wide[k] for k in ARMS_W}, open(ROOT / f"ov2_history_{WIDE_TAG}.json", "w"))
print("WIDE TRAINING DONE", flush=True)
# evaluate with the same cell as the main run
models_ov2, OV2_TAG, ARMS = models_wide, WIDE_TAG, ARMS_W
exec(open(OV2 + 'c3_eval.py').read())
print("WIDE ALL DONE", flush=True)
