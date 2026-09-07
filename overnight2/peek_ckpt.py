"""Print the probe history stored in the OV2 checkpoints on Drive (works while training runs).
usage: peek_ckpt.py [TAG] [N_LAST]          full view
       peek_ckpt.py [TAG] --compact         one line per arm (for monitors)"""
import sys, pathlib, torch
DRIVE = pathlib.Path.home() / "Library/CloudStorage/GoogleDrive-naghavnarna@gmail.com/My Drive/lisa_rtm"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
TAG = args[0] if args else "OV2_es"
N_LAST = int(args[1]) if len(args) > 1 else 3
COMPACT = "--compact" in sys.argv

def load_all(tag):
    out = []
    for p in sorted((DRIVE / "checkpoints" / tag).glob("*.pt")):
        try:
            out.append((p.stem, torch.load(p, map_location="cpu", weights_only=False)))
        except Exception as e:
            print(f"{p.stem}: unreadable ({e})", file=sys.stderr)
    return out

for name, ck in load_all(TAG):
    h = ck["history"]
    if COMPACT:
        print(f"{name}@{ck['step']} SNR0 {h['snr0'][-1]:.2f} def0 {h['def0'][-1]:+.2f} SNR1 {h['snr1'][-1]:.2f} "
              f"def1 {h['def1'][-1]:+.2f} wave {h['wave'][-1]:.4f} spread {h['spread'][-1]:.4f} (naive {h['snr_naive'][-1]:.2f})")
        continue
    print(f"{name:<10} step {ck['step']:>6}  arm {ck['arm']}  (naive probe SNR {h['snr_naive'][-1]:.2f})")
    for i in range(max(0, len(h["dev_step"]) - N_LAST), len(h["dev_step"])):
        print(f"    step {h['dev_step'][i]:>6}  SNR0 {h['snr0'][i]:6.2f} def0 {h['def0'][i]:+7.2f}   SNR1 {h['snr1'][i]:6.2f} def1 {h['def1'][i]:+7.2f}")
    print(f"    last logged terms: wave {h['wave'][-1]:.5f}  spec {h['spec'][-1]:.4f}  spread {h['spread'][-1]:.5f}  lr {h['lr'][-1]:.2e}")
