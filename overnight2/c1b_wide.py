# ============================================================ OV2-1b wide-context encoder
# Test of the receptive-field hypothesis.  LISA's encoder sees 11 input samples (0.9 ms at 12 kHz), less
# than one pitch period, so coherent continuation of harmonics above 6 kHz is impossible by construction
# and the coherent fraction kappa of its high band is ~0.  LISASW keeps the paper's local encoder and adds
# a residual stack of dilated convolutions (k=3, dilations 2..64) on the 32-d latents: receptive field
# 11 + 2*(2+4+8+16+32+64) = 263 input samples = 22 ms, two to four pitch periods.  Everything else --
# decoder, coordinate scheme, noise channels, losses -- is unchanged, so any change in kappa is context.
import torch, torch.nn as nn, torch.nn.functional as F


class LISASW(LISAS):
    def __init__(self, cfg, n_noise=N_NOISE, dilations=(2, 4, 8, 16, 32, 64)):
        super().__init__(cfg, n_noise)
        C = self.dim
        self.ctx = nn.ModuleList([nn.Conv1d(C, C, 3, padding=d, dilation=d) for d in dilations])
        self.ctx_out = nn.Conv1d(C, C, 1)
        self.rf = 11 + 2 * sum(dilations)

    def encode(self, x_lo, eps=None):
        B, L = x_lo.shape
        if eps is None:
            eps = torch.zeros(B, self.n_noise, L, device=x_lo.device, dtype=x_lo.dtype)
        z = self.enc(torch.cat([x_lo.unsqueeze(1), eps], 1))             # (B, C, L)  local latents
        h = z
        for conv in self.ctx:
            h = h + F.relu(conv(h))                                      # residual dilated stack
        return (z + self.ctx_out(h)).transpose(1, 2)                     # (B, L, C)


_m = LISASW(CFG)
print(f"LISASW: {sum(p.numel() for p in _m.parameters()):,} params (LISAS {sum(p.numel() for p in LISAS(CFG).parameters()):,}), "
      f"encoder receptive field {_m.rf} input samples = {1000 * _m.rf / CFG.fs_lo:.1f} ms")
del _m
