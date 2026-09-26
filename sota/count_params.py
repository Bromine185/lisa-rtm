"""Count the released checkpoints' parameters per architecture block, from the checkpoints themselves.

    lisa_rtm_cache/sota/sotaenv/bin/python sota/count_params.py [--out sota/params.json]

A parameter here is one element of a floating-point tensor in the state dict the model's inference path
loads (sota/run_models.py loads the same files). Each block of demo/tools/sota_models.json names the
state-dict key prefixes it owns (`count`); a block with no prefixes has no weights (an STFT, a noise draw,
a post-process). Every prefix must match at least one tensor, and for a model marked `whole` every
floating tensor of the loaded dict must fall in exactly one block, so a split cannot silently drop or
double-count weights. The total is the sum of the blocks.

Output: {model: {total, blocks: {block_id: n}, checkpoint, state_dict, excluded, whole}}. Nothing is
typed in: demo/tools/add_sota.py and make_results_sota.py read the numbers from here.
"""
import argparse, glob, json, pathlib, pickle, types

import torch

REPO = pathlib.Path(__file__).resolve().parents[1]
SOTA = pathlib.Path("/Users/raghavsharma/projects/lisa-rtm/lisa_rtm_cache/sota")


class _Stub:                                            # pytorch_lightning objects in NU-Wave 2's pickled hparams
    def __init__(self, *a, **k): pass
    def __setstate__(self, st): pass


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.startswith(("pytorch_lightning", "lightning")):
            return _Stub
        return super().find_class(mod, name)


_PL = types.SimpleNamespace(Unpickler=_U, load=pickle.load, __name__="pickle")


def audiosr_path():
    hits = glob.glob(str(pathlib.Path.home() / ".cache/huggingface/hub/models--haoheliu--audiosr_speech/snapshots/*/pytorch_model.bin"))
    if not hits:
        raise FileNotFoundError("audiosr_speech checkpoint not in the Hugging Face cache")
    return pathlib.Path(hits[0])


# where each model's inference weights live, and which part of the loaded object is the state dict
CKPT = {
    "flowhigh": (lambda: SOTA / "fh_ckpt/FLowHigh_indep_adaptive_400k.pt", "model", None,
                 "the whole `model` dict: run_models.py load_state_dict()s it, BigVGAN included"),
    "apbwe": (lambda: SOTA / "apbwe_ckpt/12kto48k/g_12kto48k", "generator", None, "the generator; the discriminators are training-time"),
    "nuwave2": (lambda: SOTA / "nuwave2_ckpt.ckpt", "state_dict", _PL, "`state_dict` under model.*: what run_models.py loads"),
    "audiosr": (audiosr_path, None, None,
                "the U-Net, the conditioning VAE's encoder, and the first-stage VAE decoder + vocoder; not the EMA copy, "
                "the CLAP model, the first-stage encoder (training-time; the conditioning branch holds an identical copy) or the schedule buffers"),
}


def floats(sd):
    return {k: v.numel() for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "sota/params.json"))
    ap.add_argument("--models", nargs="*", default=list(CKPT))
    a = ap.parse_args()
    facts = json.loads((REPO / "demo/tools/sota_models.json").read_text())["models"]
    out_p = pathlib.Path(a.out)
    out = json.loads(out_p.read_text()) if out_p.exists() else {}
    for m in a.models:
        path_of, key, pm, what = CKPT[m]
        p = path_of()
        ck = torch.load(p, map_location="cpu", weights_only=False, **({"pickle_module": pm} if pm else {}))
        sd = ck[key] if key else ck.get("state_dict", ck)
        fl = floats(sd)
        owner, blocks = {}, {}
        for b in facts[m]["blocks"]:
            n = 0
            for pre in b.get("count", []):
                hit = [k for k in fl if k == pre or k.startswith(pre + ".")]
                if not hit:
                    raise SystemExit(f"{m}/{b['id']}: prefix {pre!r} matches no tensor")
                for k in hit:
                    if k in owner:
                        raise SystemExit(f"{m}: {k} counted in both {owner[k]} and {b['id']}")
                    owner[k] = b["id"]; n += fl[k]
            blocks[b["id"]] = n
        whole = bool(facts[m].get("count_whole"))
        left = [k for k in fl if k not in owner]
        if whole and left:
            raise SystemExit(f"{m}: {len(left)} tensors in no block, e.g. {left[:3]}")
        out[m] = {"total": sum(blocks.values()), "blocks": blocks, "checkpoint": p.name, "state_dict": what,
                  "excluded": sum(fl[k] for k in left), "whole": whole}
        print(f"{m}: {out[m]['total']:,}  " + "  ".join(f"{k} {v:,}" for k, v in blocks.items()) + f"  | not counted {out[m]['excluded']:,}", flush=True)
    out_p.write_text(json.dumps(out, indent=1))
    print("wrote", out_p)


if __name__ == "__main__":
    main()
