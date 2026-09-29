"""Shared paths and file checks for the optional cache-regeneration code."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repository root
DATA = ROOT / "data"
TRAINING = ROOT / "training"
DEFAULT_OUT = TRAINING / "output"
DOWNLOADS = DEFAULT_OUT / "downloads"


def output_dir(path: str | Path) -> Path:
    """Resolve and create an output directory, refusing anything inside data/."""
    out = Path(path).resolve()
    data = DATA.resolve()
    if out == data or data in out.parents:
        raise SystemExit(f"refusing to write inside {data}; choose another --out directory")
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_source(path: str | Path, expected: str, skip: bool = False) -> Path:
    """Check that a downloaded source file exists and has the expected SHA-256."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"source file not found: {path} (see training/README.md for the download)")
    if not skip:
        got = sha256(path)
        if got != expected:
            raise SystemExit(f"SHA-256 mismatch for {path}: got {got}, expected {expected}. "
                             "Use --skip-digest to proceed anyway; results may then differ.")
    return path
