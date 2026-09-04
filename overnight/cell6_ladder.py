# ============================================================ XL-6 ladder on the chosen arm
# Pick the arm: the faithful one if it clears gate 2a; else the relu arm that clears it with the
# most headroom; else the faithful one anyway (the gates will say so).
def clears(k):
    r = RES[k]["ours120"]
    return r["snr"] > r["snr_naive"]
relu_arms = [k for k in ARM_ORDER if k.startswith("relu")]
if clears("relu_l1e-3"):
    LADDER_ARM = "relu_l1e-3"
else:
    ok = [k for k in relu_arms if clears(k)]
    LADDER_ARM = min(ok, key=lambda k: RES[k]["ours120"]["deficit"]) if ok else "relu_l1e-3"
print("ladder arm:", LADDER_ARM, "| clears gate 2a:", clears(LADDER_ARM),
      "| deficit", round(RES[LADDER_ARM]["ours120"]["deficit"], 2), "dB")

# hand the notebook's globals to §9-§15
model = models_xl[LADDER_ARM]; model.eval()
CKPT_PATH = CKPT / XL_TAG / f"{LADDER_ARM}.pt"
CFG = dataclasses.replace(CFG, name=f"XL_{LADDER_ARM}",
                          train_speakers=tuple(MANIFEST["train_speakers"]),
                          test_speakers=tuple(MANIFEST["test_speakers"]))
RUN = CKPT / XL_TAG
train_utts_full, train_spk_full = train_utts, train_spk
step_ = max(1, len(train_utts_full) // 200)
train_utts = train_utts_full[::step_][:200]
train_spk  = train_spk_full[::step_][:200]
print(f"fit subset: {len(train_utts)} utts from {len(set(train_spk))} speakers")

# run the saved notebook's §9..§15 sources in this kernel (same pattern as validate.py)
nb = json.loads((ROOT / "lisa_rtm.ipynb").read_text())
code_cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
markers = [("gate2", "pre-flight: reload what cell order"), ("ladder", "HI = slice(CFG.k_cut"),
           ("controls", "control 1: classical"), ("stochastic", "def fit_conditional_power"),
           ("latency", "def bench("), ("listen", "from IPython.display import Audio"),
           ("results", "rows = sorted(RESULTS.items()")]
for name, mark in markers:
    src = next(s for s in code_cells if mark in s)
    print(); print("=" * 30, name, "=" * 30, flush=True)
    exec(compile(src, f"<saved-notebook {name}>", "exec"), globals())
print("LADDER DONE", LADDER_ARM)
