"""Run a released SOTA checkpoint on sota/make_inputs.py's inputs, M seeded draws per utterance.

    sotaenv/bin/python sota/run_models.py --model flowhigh|nuwave2|apbwe|audiosr --set core --seeds 8 [--device mps]

Runs in the models' own environment (cache/sota/sotaenv: torch 2.2.1, numpy 1.23, librosa 0.9.2),
not the project venv. Each model is driven through its OWN inference path, re-typed here only so the seed can
be set per draw and the loop can stay in one process; nothing numeric is changed. What each one does:

  flowhigh  their inference.py: resample_poly 12k->48k, peak-normalise, cfm_wrapper.sample(time_steps=1,
            independent_cfm_adaptive, std_2=1.0), PostProcessing (STFT low-band replacement, peak 0.99).
            NOT stochastic as shipped: inference.py passes std_2=1. but not std_1, and sample() then resets
            BOTH to (1, sigma=1e-4), so y0 = cond + 1e-4 N(0, I) and draws differ by ~1e-5 relative.
  flowhigh_std1  the same, with std_1=std_2=1 passed, i.e. y0 = cond + N(0, I): the prior their training
            path (independent_cfm_adaptive, sigma_t = 1 at t = 0) actually starts from.
  nuwave2   their inference.py --gt path from the input on: resample_poly 12k->48k, 8-step DDIM with their
            infer_schedule, band mask up to 6 kHz. Stochastic: the starting noise. Loaded without
            pytorch_lightning (the Diffusion module and its weights only), since PL 1.2 does not install.
  apbwe     their inference_48k.py: torchaudio resample 12k->48k, amp/phase STFT, model, iSTFT.
            Deterministic: one draw regardless of --seeds.
  apbwe_sinc the same model on ITS OWN test input (torchaudio sinc 48k -> 12k -> 48k from the truth), not the
            Chebyshev file: the protocol its paper reports, to separate model from input-filter shift.
  audiosr   audiosr.super_resolution(model_name="speech", ddim_steps=50, guidance_scale=3.5), the CLI
            defaults, on the input upsampled to 48 kHz by torchaudio (what their read_wav_file does).
            Stochastic: seed_everything(seed) before each draw.

Output: <work>/<set>/out/<model>/<utt>_s<k>.npy, float32 at 48 kHz, cropped to the truth's length.
Existing outputs are skipped, so a killed run resumes.
"""
import argparse, json, os, pathlib, sys, time
import numpy as np
import soundfile as sf
import scipy.signal as sps

RUN_CTX = {}
SOTA = pathlib.Path(os.environ.get("BWE_CACHE", str(pathlib.Path(__file__).resolve().parents[1] / "cache")) + "/sota")
FS = 48000


def flowhigh(dev, std1=False):
    fh = SOTA / "FLowHigh_code"; sys.path.insert(0, str(fh)); os.chdir(fh)
    os.environ["FLOWHIGH_DEVICE"] = dev
    import torch
    from cfm_superresolution import MelVoco, FLowHigh, ConditionalFlowMatcherWrapper
    from postprocessing import PostProcessing
    D = torch.device(dev)
    enc = MelVoco(n_mels=256, sampling_rate=FS, f_max=24000, n_fft=2048, win_length=2048, hop_length=480,
                  vocoder="bigvgan", vocoder_config=str(fh / "vocoder/BIGVGAN/config/bigvgan_48khz_256band_config.json"),
                  vocoder_path=str(fh / "vocoder/BIGVGAN/checkpoint/g_48_00850000"))
    ck = torch.load(SOTA / "fh_ckpt/FLowHigh_indep_adaptive_400k.pt", map_location="cpu")
    gen = FLowHigh(dim_in=enc.n_mels, audio_enc_dec=enc, depth=2, dim_head=64, heads=16, architecture="transformer")
    w = ConditionalFlowMatcherWrapper(flowhigh=gen, cfm_method="independent_cfm_adaptive", torchdiffeq_ode_method="euler", sigma=1e-4)
    w.load_state_dict(ck["model"]); gen.to(D).eval(); w.to(D).eval()
    pp = PostProcessing(0)

    def run(x_lo, n, seed):
        cond = sps.resample_poly(x_lo, FS, FS // 4); cond = cond / np.max(np.abs(cond))
        cond = torch.tensor(cond, dtype=torch.float32)[None].to(D)
        torch.manual_seed(seed)
        with torch.no_grad():
            if std1:   # both stds given, so sample() keeps them: the prior the model was trained from
                hr = w.sample(cond=cond, time_steps=1, cfm_method="independent_cfm_adaptive", std_1=1., std_2=1.).squeeze(1)
            else:      # their call, verbatim: std_1 is None, so sample() resets std_2 to sigma = 1e-4
                hr = w.sample(cond=cond, time_steps=1, cfm_method="independent_cfm_adaptive", std_2=1.).squeeze(1)
            hr = pp.post_processing(hr, cond, cond.size(-1))
        return hr.cpu().squeeze().clamp(-1, 1).numpy()
    return run, True


def nuwave2(dev):
    nw = SOTA / "nuwave2"; sys.path.insert(0, str(nw)); os.chdir(nw)
    import torch
    from omegaconf import OmegaConf as OC
    from diffusion import Diffusion
    hp = OC.load(nw / "hparameter.yaml")
    D = torch.device(dev)
    net = Diffusion(hp)
    import pickle, types

    class _Stub:                                                     # PL objects in the pickle's hparams
        def __init__(self, *a, **k): pass
        def __setstate__(self, st): pass

    class _U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(("pytorch_lightning", "lightning")):
                return _Stub
            return super().find_class(mod, name)
    pm = types.SimpleNamespace(Unpickler=_U, load=pickle.load, __name__="pickle")
    sd = torch.load(SOTA / "nuwave2_ckpt.ckpt", map_location="cpu", pickle_module=pm)["state_dict"]
    net.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
    net.to(D).eval()
    sched = eval(hp.dpm.infer_schedule)
    fft_size = hp.audio.filter_length // 2 + 1

    def run(x_lo, n, seed):
        wl = sps.resample_poly(x_lo, FS, FS // 4)[:n]
        wl = wl[: len(wl) - len(wl) % hp.audio.hop_length]           # their inference.py: a multiple of the hop
        g = np.max(np.abs(wl)); wl = wl / g                          # their test wavs are peak 1
        band = torch.zeros(fft_size, dtype=torch.int64); band[: int((6000 / 24000) * fft_size)] = 1
        wl = torch.from_numpy(wl.copy()).float()[None].to(D); band = band[None].to(D)
        torch.manual_seed(seed)
        with torch.no_grad():                                         # lightning_model.NuWave2.inference, verbatim
            sig = torch.randn(wl.shape, dtype=wl.dtype, device="cpu").to(D)
            for i in range(8):
                lt = sched[i] * torch.ones(1, device=D)
                ls = (hp.logsnr.logsnr_max if i == 7 else sched[i + 1]) * torch.ones(1, device=D)
                sig, _ = net.denoise_ddim(sig, wl, band, lt, ls)
            out = torch.clamp(sig, min=-1, max=1 - torch.finfo(torch.float16).eps)
        return out[0].cpu().numpy() * g
    return run, True


def apbwe(dev, own_input=False):
    ap_ = SOTA / "AP-BWE"; sys.path.insert(0, str(ap_)); os.chdir(ap_)
    import torch, torchaudio.functional as aF
    from env import AttrDict
    from datasets.dataset import amp_pha_stft, amp_pha_istft
    from models.model import APNet_BWE_Model
    cdir = SOTA / "apbwe_ckpt/12kto48k"
    h = AttrDict(json.load(open(cdir / "config.json")))
    D = torch.device(dev)
    m = APNet_BWE_Model(h).to(D)
    m.load_state_dict(torch.load(cdir / "g_12kto48k", map_location="cpu")["generator"]); m.eval()

    def run(x_lo, n, seed):
        # AP-BWE is NOT level invariant (log-amplitude in, no normalisation): on 16 core utterances its LSD is
        # 0.776 at the corpus's stored level (mean peak 0.33, what it trained and was tested on) and 0.943 at
        # peak 0.95. So it is fed at the stored level and its output scaled back by the same factor.
        k = RUN_CTX["raw_peak"] / 0.95
        if own_input:     # their inference_48k.py verbatim: torchaudio sinc 48k -> 12k -> 48k from the truth
            hr = torch.tensor(RUN_CTX["truth"] * k, dtype=torch.float32)[None].to(D)
            lr = aF.resample(aF.resample(hr, orig_freq=48000, new_freq=12000), orig_freq=12000, new_freq=48000)[:, :n]
        else:
            a = torch.tensor(x_lo * k, dtype=torch.float32)[None].to(D)
            lr = aF.resample(a, orig_freq=12000, new_freq=48000)[:, :n]
        with torch.no_grad():
            amp, pha, _ = amp_pha_stft(lr, h.n_fft, h.hop_size, h.win_size)
            A, P, _ = m(amp, pha)
            out = amp_pha_istft(A, P, h.n_fft, h.hop_size, h.win_size)
        return out.squeeze().cpu().numpy() / k
    return run, False


def audiosr(dev):
    import torch, tempfile, audiosr as A
    model = A.build_model(model_name="speech", device=dev)
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="audiosr_in_"))

    def run(x_lo, n, seed):
        f = tmp / "in.wav"; sf.write(str(f), x_lo.astype(np.float32), 12000, subtype="FLOAT")
        w = A.super_resolution(model, str(f), seed=seed, guidance_scale=3.5, ddim_steps=50, latent_t_per_second=12.8)
        return np.asarray(w).squeeze()
    return run, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["flowhigh", "flowhigh_std1", "nuwave2", "apbwe", "apbwe_sinc", "audiosr"])
    ap.add_argument("--set", default="core")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--work", default=str(SOTA / "work"))
    a = ap.parse_args()
    ws = pathlib.Path(a.work) / a.set
    utts = json.load(open(ws / "utts.json"))["utts"][: a.n]
    out = ws / "out" / a.model; out.mkdir(parents=True, exist_ok=True)
    run, stochastic = {"flowhigh": flowhigh, "flowhigh_std1": lambda d: flowhigh(d, std1=True), "nuwave2": nuwave2, "apbwe": apbwe, "apbwe_sinc": lambda d: apbwe(d, own_input=True), "audiosr": audiosr}[a.model](a.device)
    seeds = range(a.seeds if stochastic else 1)
    t0, audio_s, done = time.time(), 0.0, 0
    for i, u in enumerate(utts):
        x_lo, sr = sf.read(str(ws / "lo12k" / f"{u['utt']}.wav"), dtype="float64")
        RUN_CTX["truth"] = np.load(ws / "truth" / f"{u['utt']}.npy")
        RUN_CTX["raw_peak"] = u.get("raw_peak", 0.95)
        n = int(round(u["seconds"] * FS))
        for s in seeds:
            f = out / f"{u['utt']}_s{s}.npy"
            if f.exists():
                continue
            t1 = time.time()
            w = run(x_lo, n, s)
            w = np.pad(w, (0, max(0, n - len(w))))[:n].astype(np.float32)
            np.save(f, w); audio_s += n / FS; done += 1
            print(f"  [{time.time()-t0:6.0f}s] {a.model} {i+1}/{len(utts)} {u['utt']} s{s} {n/FS:.2f}s "
                  f"in {time.time()-t1:.1f}s", flush=True)
    wall = time.time() - t0
    if not done:   # every draw was already on disk: keep the _run.json of the run that made them
        print(f"{a.model}: nothing to run, _run.json kept", flush=True)
        return
    meta = {"model": a.model, "device": a.device, "seeds": len(seeds), "stochastic": stochastic,
            "wall_s": wall, "audio_s": audio_s, "rtf": wall / max(audio_s, 1e-9), "draws_run": done,
            "machine": machine()}
    json.dump(meta, open(out / "_run.json", "w"), indent=1)
    print(meta, flush=True)


def machine():
    """The CPU this ran on, as fast/bench_latency.py names it."""
    import platform, subprocess
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return platform.processor() or None


if __name__ == "__main__":
    main()
