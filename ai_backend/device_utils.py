"""Shared device-selection helper: CPU by default, AMD GPU via DirectML only
if explicitly requested, then CUDA, then MPS. Pattern originally ported from
the user's own working setup in
D:\\ΠΜΣ\\Βαθιά Μάθηση\\Εργασίες εξαμήνου\\2η Άσκηση\\25118.py, used across
every Track A training script so device selection is defined once, not
duplicated per file.

DirectML defaults to off, not on: at this project's fixed batch_size=64 (the
accuracy-validated config, see train.py::train_model's docstring -- larger
batches measure worse on the real chained-trajectory metric, so batch=64 is
not going away), DirectML's per-batch dispatch overhead makes it
substantially slower than plain CPU in real training loops on this machine's
AMD Radeon RX 9070 -- an LSTM epoch measured ~249s/epoch on DirectML vs.
~28s/epoch on CPU for the identical training loop, a ~9x gap, far worse than
the raw-throughput micro-benchmark alone suggested (1692 vs. 2068 rows/sec,
see models/ANALYSIS.md's Infrastructure note). Pass prefer_directml=True to
opt back into GPU, e.g. when deliberately exploring the batch=256+ regime
where DirectML does win.
"""
import torch


def get_device(verbose: bool = True, prefer_directml: bool = False):
    if prefer_directml:
        try:
            import torch_directml
            device = torch_directml.device(0)
            name = torch_directml.device_name(0)
            if verbose:
                print(f"Device: DirectML - {name}")
            return device
        except Exception as e:
            if verbose:
                print(f"DirectML not available ({type(e).__name__}: {e}), falling back")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        "cpu"
    )
    if verbose:
        print(f"Device: {device}")
    return device


if __name__ == '__main__':
    d = get_device()
    x = torch.randn(4, 4).to(d)
    y = x @ x
    print(f"Smoke test matmul on {d}: shape {y.shape}, finite: {torch.isfinite(y).all().item()}")
    assert torch.isfinite(y).all()
    print("Device helper OK.")
