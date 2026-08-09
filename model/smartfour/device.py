"""Device selection for training and inference: CUDA > MPS > CPU.

`resolve_device` picks the best available accelerator (a CUDA GPU, then Apple
MPS on macOS) with CPU as the universal fallback, so the same code runs
unchanged on a GPU box, a Mac, or a CPU-only host. An explicit `preferred`
device is always honored — it bypasses availability checks and lets users
force CPU even when a GPU is present.
"""

import torch


def resolve_device(preferred: str | None = None) -> torch.device:
    """Return the device to run the net on.

    Auto mode (preferred=None): cuda if any GPU is visible, else mps on
    Apple Silicon, else cpu. A non-None preferred value is returned as-is.
    """
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device) -> str:
    """Human-readable device label for startup banners, e.g. 'cuda (NVIDIA ...)'."""
    d = torch.device(device)
    if d.type == "cuda":
        name = torch.cuda.get_device_name(d.index or 0)
        return f"cuda ({name})"
    if d.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def state_to_cpu(obj):
    """Deep-copy net/optimizer state onto CPU, detached — device-portable
    checkpoints and IPC. Recurses through dicts/lists; anything exposing
    detach()/cpu() (i.e. tensors) is moved, other values pass through.
    Optimizer state is nested (state dict + param_groups), so this must walk.
    """
    if isinstance(obj, dict):
        return {k: state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [state_to_cpu(v) for v in obj]
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):
        return obj.detach().cpu()
    return obj
