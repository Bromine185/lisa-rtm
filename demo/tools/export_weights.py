#!/usr/bin/env python
"""Export LISAS / LISASD checkpoints to flat Float32 little-endian blobs for demo/engine.js.

    venv/bin/python demo/tools/export_weights.py <ckpt.pt> [more.pt ...] --out demo/assets/weights

One <arm>.bin per checkpoint (arm = file stem with any _stepNNNN suffix removed) and one entry per
arm merged into <out>/manifest.json.  Existing entries for other arms are kept.

Blob layout.  Every tensor of the state dict, in state-dict order, as float32 little-endian, each in
PyTorch's own row-major memory order:
    enc.<2l>.weight   (c_out, c_in, k)      Conv1d, cross-correlation, zero 'same' padding k//2
    enc.<2l>.bias     (c_out,)
    dec.net.<2l>.weight (out, in)           Linear, y = W x + b
    dec.net.<2l>.bias   (out,)
The decoder's first Linear takes [c, z_{i-1} (C), z_i (C), z_{i+1} (C), eps_dec (n_dec)] in that column
order (overnight3/e1_model.py, _decode_gather), so its weight is (H, 1 + 3C + n_dec).  The manifest
records the byte offset and byte length of every tensor so a reader never has to know the layout.
"""
import argparse, json, pathlib, re, sys

import numpy as np
import torch

STEP_RE = re.compile(r"_step\d+$")


def arm_name(path):
    return STEP_RE.sub("", pathlib.Path(path).stem)


def export_one(ckpt_path, out_dir, R):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    arm = arm_name(ckpt_path)
    kind = ck["arm"][0] if isinstance(ck.get("arm"), (tuple, list)) else str(ck.get("arm", ""))
    lam = ck["arm"][1] if isinstance(ck.get("arm"), (tuple, list)) and len(ck["arm"]) > 1 else None
    cls = ck.get("cls", "LISAS")

    # ---- geometry from the tensors themselves (the checkpoint's own fields are cross-checked) ----
    enc_keys = sorted((k for k in sd if k.startswith("enc.") and k.endswith(".weight")),
                      key=lambda k: int(k.split(".")[1]))
    enc_channels = [int(sd[k].shape[0]) for k in enc_keys]
    enc_kernels = [int(sd[k].shape[2]) for k in enc_keys]
    n_in = int(sd[enc_keys[0]].shape[1])
    n_noise = n_in - 1
    C = enc_channels[-1]
    dec_keys = sorted((k for k in sd if k.startswith("dec.net.") and k.endswith(".weight")),
                      key=lambda k: int(k.split(".")[2]))
    W1 = sd[dec_keys[0]]
    H = int(W1.shape[0])
    n_dec = int(W1.shape[1]) - 1 - 3 * C
    dec_layers = len(dec_keys)
    if n_dec < 0:
        sys.exit(f"{ckpt_path}: first decoder Linear has {W1.shape[1]} inputs < 1 + 3*{C}")
    if "n_noise" in ck and int(ck["n_noise"]) != n_noise:
        sys.exit(f"{ckpt_path}: ck['n_noise']={ck['n_noise']} but enc.0 has {n_in} input channels")
    if "n_dec" in ck and int(ck["n_dec"]) != n_dec:
        sys.exit(f"{ckpt_path}: ck['n_dec']={ck['n_dec']} but dec.net.0 has {W1.shape[1]} inputs")
    if n_dec > 0 and cls == "LISAS":
        cls = "LISASD"
    for k in dec_keys[1:-1]:
        assert tuple(sd[k].shape) == (H, H), (k, sd[k].shape)
    assert tuple(sd[dec_keys[-1]].shape) == (1, H), sd[dec_keys[-1]].shape

    # ---- write the blob ----
    layers, chunks, off = [], [], 0
    for name, t in sd.items():
        a = t.detach().cpu().contiguous().numpy().astype("<f4")
        b = a.tobytes(order="C")
        layers.append({"name": name, "shape": [int(s) for s in t.shape], "offset": off,
                       "length": len(b), "numel": int(a.size)})
        chunks.append(b)
        off += len(b)
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{arm}.bin"
    bin_path.write_bytes(b"".join(chunks))
    param_count = sum(l["numel"] for l in layers)

    entry = {
        "file": bin_path.name,
        "cls": cls,
        "n_noise": n_noise,
        "n_dec": n_dec,
        "det": bool(kind.startswith("det")),
        "kind": kind,
        "lam": lam,
        "step": int(ck.get("step", 0)),
        "param_count": param_count,
        "bytes": off,
        "enc_channels": enc_channels,
        "enc_kernels": enc_kernels,
        "latent": C,
        "dec_hidden": H,
        "dec_layers": dec_layers,
        "R": R,
        "fs_lo": 48000 // R,
        "fs_hi": 48000,
        "dec_input_order": "[c, z_{i-1}, z_i, z_{i+1}, eps_dec]",
        "layers": layers,
    }
    return arm, entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--out", required=True, help="directory for <arm>.bin and manifest.json")
    ap.add_argument("--R", type=int, default=4, help="training upsample factor (fs_hi / fs_lo)")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    man_path = out_dir / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    manifest.setdefault("format", "float32 little-endian; tensors concatenated in the listed order, "
                                  "each in PyTorch row-major layout; offset and length are bytes")
    manifest.setdefault("arms", {})
    for p in args.ckpts:
        arm, entry = export_one(pathlib.Path(p), out_dir, args.R)
        manifest["arms"][arm] = entry
        print(f"{arm:24s} {entry['cls']:7s} n_noise {entry['n_noise']} n_dec {entry['n_dec']} "
              f"det {str(entry['det']):5s} step {entry['step']:>6d} params {entry['param_count']:,} "
              f"-> {entry['file']} ({entry['bytes']:,} B)")
    man_path.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"manifest: {man_path} ({len(manifest['arms'])} arm(s))")


if __name__ == "__main__":
    main()
