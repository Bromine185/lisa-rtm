"""Train one ENV arm (fast/arms_env.py) with fast/train_arm.py's resumable loop, paired with OV50.

    python fast/train_env.py --arm es_ctx_env_l0.1 --corpus /content/corpus --out /content/drive/MyDrive/lisa_rtm/env_run
    python fast/train_env.py --arm es_env_l0.1 --smoke                       # CPU, synthetic, a few steps
    python fast/train_env.py --arm es_ctx_env_l0.1 --out <run> --export-only  # resume ckpt -> load_arm blob

What is reused, and from where: the boot (train_arm.boot), the corpus (train_arm.StagedCorpus), the loop,
the checkpoint format and the resume (train_arm.train), the batch stream (run_contract.BATCH_TAG, the same
constant OV50 used, so these arms see the same 104,950 batches in the same order as es_dec_erb_l0.1). The
init is the control's too: seed_as_control() below. So each ENV arm and the control share the batches and
the step-0 function; they differ in the one change the arm makes and in the realised noise draws.  What is new is exec'd from fast/arms_env.py into the same namespace: the LISASDW
class, the ES_env term, and the model/stack builders that train_arm.train() looks up through G.

The three arms are registered into run_contract.ARMS at runtime (train_arm.train reads that dict), so the
OV50 table itself is not edited.  After a finished run (and after --smoke) the inference checkpoint that
load_arm() reads, the blob fast/convert_ckpt.py describes, is written to <out>/ckpt/<arm>.pt; --export-only
re-exports it from the resume checkpoint.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast import train_arm as TA                                    # noqa: E402
from fast.run_contract import ARMS, BATCH, RUN_TAG, SEG_HI, STEPS, preflight   # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
ENV_ARMS = {                        # mirrors ARMS_ENV in fast/arms_env.py (asserted equal after boot)
    "es_env_l0.1":     ("es_marg_erb_env", 1e-1, "LISASD"),
    "es_ctx_erb_l0.1": ("es_marg_erb",     1e-1, "LISASDW"),
    "es_ctx_env_l0.1": ("es_marg_erb_env", 1e-1, "LISASDW"),
}


CONTROL = "es_dec_erb_l0.1"           # the OV50 arm every ENV arm is compared against


def seed_as_control(arm, seed=None):
    """Every ENV arm starts from the CONTROL's seed, not its own. train_arm.train() seeds torch with
    arm_seed(arm) and then builds the models, so with the control's seed LISAS is initialised exactly as
    OV50's es_dec_erb_l0.1 was; es_env_l0.1 (LISASD) then matches it tensor for tensor, and LISASDW copies the
    same shared weights (copy_shared) with ctx_out at zero, so all three arms are the control's step-0
    FUNCTION. What cannot be paired: the noise and jitter draws during training (they depend on the GPU
    model, see fast/train_arm.py) and, for LISASDW, the extra RNG its context convs consume."""
    return _orig_arm_seed(CONTROL) if arm in ENV_ARMS else _orig_arm_seed(arm)


_orig_arm_seed = TA.arm_seed


def boot_env(smoke=False, device=None, root=None):
    TA.arm_seed = seed_as_control                                  # train_arm.train() looks it up at call time
    G = TA.boot(smoke=smoke, device=device, root=root)
    exec(compile((REPO / "fast/arms_env.py").read_text(), "<fast/arms_env.py>", "exec"), G)
    assert G["ARMS_ENV"] == ENV_ARMS, "fast/arms_env.py and fast/train_env.py disagree on the arms"
    ARMS.update(ENV_ARMS)                                            # train_arm.train reads run_contract.ARMS
    G["_make_models"], G["build_stacks"] = G["make_models_env"], G["build_stacks_env"]
    return G


def export(G, arm, out):
    """Resume checkpoint (stacked params) -> the plain-module blob overnight2/c1_model.py::load_arm reads."""
    import torch
    out = pathlib.Path(out)
    ck = torch.load(out / f"{arm}.pt", map_location="cpu", weights_only=False)
    kind, lam, cls = ENV_ARMS[arm]
    names, specs, base, models = G["make_models_env"]({arm: (kind, lam, cls)})
    stack = G["build_stacks_env"](names, specs, base, G["DEVICE"])[0]
    with torch.no_grad():
        for k, v in ck["params"].items():
            stack.P[k].copy_(v.to(stack.P[k].device))
    m = stack.export(0, models[arm])
    blob = {"model": {k: v.detach().cpu() for k, v in m.state_dict().items()}, "step": ck["step"], "history": ck["hist"],
            "arm": (kind, lam, cls), "n_noise": m.n_noise, "n_dec": getattr(m, "n_dec", 0), "cls": cls,
            "batch": BATCH, "seg": SEG_HI, "tag": ck.get("run_tag", RUN_TAG),
            "note": "ENV arm (fast/arms_env.py); load with that file exec'd so the class resolves"}
    (out / "ckpt").mkdir(exist_ok=True)
    dst = out / "ckpt" / f"{arm}.pt"
    torch.save(blob, dst)
    print(f"exported {arm} step {ck['step']} -> {dst}", flush=True)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ENV_ARMS))
    ap.add_argument("--corpus")
    ap.add_argument("--out", default="env_run")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--ckpt-every", type=int, default=500)
    ap.add_argument("--stop-after", type=int, default=None, help="end this session at this step; resume later")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    a = ap.parse_args()

    out = pathlib.Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    G = boot_env(smoke=a.smoke, device="cpu" if a.smoke else None, root=out)
    import torch
    print(f"arm {a.arm} {ENV_ARMS[a.arm]}  device {G['DEVICE']}  torch {torch.__version__}", flush=True)
    if a.export_only:
        export(G, a.arm, out); return 0

    if a.smoke:
        v = (REPO / "validate.py").read_text()
        gen = "def synthetic_corpus" + v.split("def synthetic_corpus")[1].split("\nfailures = []")[0]
        ns = {"G": G}; exec(gen, ns); ns["synthetic_corpus"](None)
        seg = 4096
        corpus = G["HostCorpus"](G["train_utts"], G["CFG"], seg_hi=seg)
        val_corpus = G["HostCorpus"](G["test_utts"], G["CFG"], seg_hi=seg)
    else:
        corpus = TA.StagedCorpus(a.corpus, G["CFG"], device=G["DEVICE"])
        val_corpus = corpus
        print("preflight:", json.dumps(preflight(corpus.n, corpus.hours, SEG_HI)), flush=True)
        print("g3 (must equal OV50's manifest to be paired):",
              json.dumps({k: corpus.manifest["g3"][k] for k in ("corpus", "batches")}), flush=True)

    h = TA.train(G, a.arm, corpus, val_corpus, out, steps=a.steps, batch=a.batch, ckpt_every=a.ckpt_every,
                 val_every=min(a.ckpt_every, max(1, a.steps // 4)), log_every=max(1, min(25, a.steps // 8)),
                 n_val_batches=2 if a.smoke else 8, resume=not a.no_resume, stop_after=a.stop_after)
    (out / f"history_{RUN_TAG}_{a.arm}.json").write_text(json.dumps({a.arm: h}))
    last = h["val_step"][-1] if h["val_step"] else 0
    print(f"{'DONE' if last >= a.steps else 'PAUSED'} {a.arm} at step {last}/{a.steps}: final val {h['val_loss'][-1]:.6f}", flush=True)
    if last >= a.steps or a.smoke:
        export(G, a.arm, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
