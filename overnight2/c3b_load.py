# ============================================================ OV2-3b load the trained arms from Drive (no training)
models_ov2 = {}
for p in sorted((CKPT / "OV2_es").glob("*.pt")):
    models_ov2[p.stem], _ck = load_arm(p)
ARMS = {k: torch.load(CKPT / "OV2_es" / f"{k}.pt", map_location="cpu", weights_only=False)["arm"] for k in models_ov2}
OV2_TAG = "OV2_es"
print("loaded arms:", {k: (ARMS[k], models_ov2[k].tau) for k in models_ov2}, flush=True)
