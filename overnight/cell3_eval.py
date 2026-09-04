# ============================================================ XL-3 evaluation harness
import json, numpy as np, scipy.signal as sps

def evaluate_model(m, utts, n, label, cfg=None):
    cfg = cfg or CFG
    rows = []
    for y in utts[:n]:
        y = np.asarray(y, np.float64)
        yh = reconstruct(m, y, cfg)
        yn = naive_upsample(y, cfg)
        b = band_energy_ratio(y, yh, cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop, 200.0, cfg.fs_hi / 2)
        hb = b[:, 0] >= cfg.fs_lo / 2
        rows.append(dict(snr=snr_db(y, yh), snr_naive=snr_db(y, yn),
                         lsd=lsd_db(y, yh, cfg.eval_n_fft, cfg.eval_hop),
                         lsd_paper_basis=lsd_db(y, yh, 2048, 1024),
                         hb_lsd=lsd_db(y, yh, cfg.eval_n_fft, cfg.eval_hop, cfg.eval_k_cut),
                         deficit=float(b[hb, 1].mean()), baseband=float(b[~hb, 1].mean())))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["n"] = len(rows)
    print(f"{label:<28}", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in agg.items()}))
    return agg

def phase_swap(m, utts, n, label, cfg=None):
    cfg = cfg or CFG
    r = []
    for y in utts[:n]:
        y = np.asarray(y, np.float64)
        yh = reconstruct(m, y, cfg)
        k = min(len(y), len(yh)); y, yh = y[:k], yh[:k]
        N, H = cfg.eval_n_fft, cfg.eval_hop
        _, _, Y = sps.stft(y, nperseg=N, noverlap=N - H)
        _, _, P = sps.stft(yh, nperseg=N, noverlap=N - H)
        _, a = sps.istft(np.abs(P) * np.exp(1j * np.angle(Y)), nperseg=N, noverlap=N - H)
        _, b = sps.istft(np.abs(Y) * np.exp(1j * np.angle(P)), nperseg=N, noverlap=N - H)
        r.append((snr_db(y, yh), snr_db(y, a[:k]), snr_db(y, b[:k])))
    r = np.array(r).mean(0)
    print(f"{label:<28} SNR as-is {r[0]:6.2f} | pred-mag+target-phase {r[1]:6.2f} | target-mag+pred-phase {r[2]:6.2f}")
    return r

def deficit_curve(m, utts, n, cfg=None):
    cfg = cfg or CFG
    bands = np.mean([band_energy_ratio(np.asarray(y, np.float64), reconstruct(m, np.asarray(y, np.float64), cfg),
                                       cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop, 200.0, cfg.fs_hi / 2)[:, 1]
                     for y in utts[:n]], axis=0)
    centres = band_energy_ratio(np.asarray(utts[0], np.float64), reconstruct(m, np.asarray(utts[0], np.float64), cfg),
                                cfg.fs_hi, cfg.eval_n_fft, cfg.eval_hop, 200.0, cfg.fs_hi / 2)[:, 0]
    return centres, bands

def dump_weights_b64(m, label):
    '''fp16 state dict as base64 -- survives in the notebook output when Drive is not mounted.'''
    import io, base64
    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().half() for k, v in m.state_dict().items()}, buf)
    s = base64.b64encode(buf.getvalue()).decode()
    print(f"WEIGHTS_B64 {label} {len(s)}")
    print(s)
