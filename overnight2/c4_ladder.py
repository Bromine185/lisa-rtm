# ============================================================ OV2-4 post-hoc ladder + stochastic rung on the det arm
# The baseline the learned sampler has to beat: the notebook's own S rung (noise through a closed-form
# conditional transport map) on the deterministic model, scored on the same 12 held-out utterances.
LADDER_ARM = globals().get("LADDER_ARM", "det")
model = models_ov2[LADDER_ARM]; model.eval(); model.tau = 0.0
CKPT_PATH = CKPT / OV2_TAG / f"{LADDER_ARM}.pt"
CFG = dataclasses.replace(CFG, name=f"OV2_{LADDER_ARM}",
                          train_speakers=tuple(MANIFEST["train_speakers"]), test_speakers=tuple(MANIFEST["test_speakers"]))
RUN = CKPT / OV2_TAG
train_utts_full, train_spk_full = train_utts, train_spk
step_ = max(1, len(train_utts_full) // 200)
train_utts = train_utts_full[::step_][:200]
train_spk = train_spk_full[::step_][:200]
n_params = sum(p.numel() for p in model.parameters())
_code = ["".join(c["source"]) for c in json.loads((Path(OV2) / "lisa_rtm.ipynb").read_text())["cells"] if c["cell_type"] == "code"]
for name, mark in [("gate2", "pre-flight: reload what cell order"), ("ladder", "HI = slice(CFG.k_cut"),
                   ("controls", "control 1: classical"), ("stochastic", "def fit_conditional_power")]:
    src = next(s for s in _code if mark in s)
    print(); print("=" * 30, name, "=" * 30, flush=True)
    exec(compile(src, f"<saved-notebook {name}>", "exec"), globals())
LADDER = {"arm": LADDER_ARM, "results": RESULTS, "crps": CRPS}
json.dump(LADDER, open(ROOT / "ov2" / f"ladder_{OV2_TAG}_{LADDER_ARM}.json", "w"), indent=1)
train_utts, train_spk = train_utts_full, train_spk_full
print("LADDER DONE", LADDER_ARM, "CRPS", CRPS, flush=True)
