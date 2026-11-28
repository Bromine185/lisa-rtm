"""Make NU-Wave 2's model.py run under torch 2, with the same numbers on CPU and CUDA.

    python sota/patch_nuwave2.py <path to nuwave2 checkout>

Its FourierUnit builds a real [..., 2] spectrum (stft return_complex=False) and calls torch.istft on it, which
torch 2 rejects; the patch reads the tensor as complex (view_as_complex). The spectrum is a network output, so
its DC and Nyquist bins can carry an imaginary part that no real signal has. The one-sided inverse FFT then
depends on the backend: pocketfft (CPU) drops those imaginary parts, cuFFT does not. With the CPU semantics the
model reproduces NU-Wave 2's published LSD (0.989 here against their 0.94); with cuFFT's it scores about 0.29
worse and fills the high band 8 dB louder. The patch drops the two imaginary parts explicitly, which is what the
CPU already does, so CPU outputs are unchanged and CUDA matches them. Idempotent.
"""
import pathlib, sys

OLD = "output = torch.istft(ffted, "
OLD1 = "output = torch.istft(torch.view_as_complex(ffted.contiguous()), "
NEW = "output = torch.istft(_hermitian(ffted), "
HELPER = '''

def _hermitian(ffted):
    """[..., F, T, 2] real layout -> complex, imaginary parts of the DC and Nyquist bins dropped (as pocketfft does)."""
    X = torch.view_as_complex(ffted.contiguous())
    return torch.cat([X[:, :1].real.to(X.dtype), X[:, 1:-1], X[:, -1:].real.to(X.dtype)], 1)
'''

p = pathlib.Path(sys.argv[1]) / "model.py"; t = p.read_text()
if "_hermitian(" not in t:
    t = t.replace(OLD1, NEW).replace(OLD, NEW)
    assert t.count(NEW) == 1, "istft call not found"
    head, sep, rest = t.partition("\nclass ")
    t = head.rstrip("\n") + "\n" + HELPER + "\n" + sep.lstrip("\n") + rest if sep else t + HELPER
    p.write_text(t)
print("nuwave2 model.py patched:", "_hermitian(" in p.read_text())
