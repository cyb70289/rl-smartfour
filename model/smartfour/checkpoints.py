"""Versioned inference checkpoints: best{n}.pt, biggest n = strongest.

Every arena promotion writes a new best{n}.pt (n = previous max + 1), so the
biggest n is always the strongest model. The UI bridge lists them (biggest n
first) and defaults to the biggest one. Plain best.pt and any other .pt file
are not part of this scheme and are ignored.
"""

import re
from pathlib import Path

_BEST_RE = re.compile(r"^best(\d+)\.pt$")


def best_versions(checkpoint_dir) -> list[int]:
    """Version numbers n of every best{n}.pt file in the dir, ascending."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.is_dir():
        return []
    versions = []
    for path in checkpoint_dir.iterdir():
        m = _BEST_RE.match(path.name)
        if m and path.is_file():
            versions.append(int(m.group(1)))
    return sorted(versions)


def latest_best(checkpoint_dir) -> Path | None:
    """The biggest best{n}.pt in the dir, or None when none exists."""
    versions = best_versions(checkpoint_dir)
    if not versions:
        return None
    return Path(checkpoint_dir) / f"best{versions[-1]}.pt"
