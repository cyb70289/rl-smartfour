"""Device resolution for training and the inference server.

`auto` picks the first accelerator available (cuda -> mps -> cpu). Requesting
an unavailable device is a hard error — never a silent fallback, which would
poison benchmarks and mask real setup problems.
"""

import torch

VALID_DEVICES = ("auto", "cpu", "mps", "cuda")


def resolve_device(name: str = "auto") -> str:
    """Map a device name to a concrete torch device string.

    auto: cuda when available, else mps, else cpu. cpu/mps/cuda are returned
    unchanged after an availability check (hard error when missing).
    """
    if name not in VALID_DEVICES:
        raise ValueError(
            f"unknown device {name!r}; expected one of {', '.join(VALID_DEVICES)}"
        )
    if name == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but MPS is not available")
    return name


def synchronize(device: str) -> None:
    """Block until all work on `device` has completed (no-op for cpu)."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
