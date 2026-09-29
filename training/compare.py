#!/usr/bin/env python3
"""Compare regenerated cache files with the shipped ones in data/, array by array.

Usage:
    python training/compare.py training/output            # every regenerated cache with a data/ counterpart
    python training/compare.py training/output/har/state data/har/state

With one argument, every ``*.npz`` below the given directory whose path
relative to it also exists under data/ is compared (prepared inputs and model
checkpoints have no counterpart and are skipped). With two arguments, the
files of the first directory are compared with the same names in the second.
Integer and boolean arrays must be identical; floating-point arrays must agree
within ``--atol`` (default 0, i.e. exactly). Exit status is 0 only if every
compared file matches and at least one file was compared.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def compare_file(new: Path, ref: Path, atol: float) -> list[str]:
    problems = []
    with np.load(new) as a, np.load(ref) as b:
        keys_a, keys_b = set(a.files), set(b.files)
        if keys_a != keys_b:
            if keys_b - keys_a:
                problems.append(f"missing arrays: {sorted(keys_b - keys_a)}")
            if keys_a - keys_b:
                problems.append(f"extra arrays: {sorted(keys_a - keys_b)}")
        for key in sorted(keys_a & keys_b):
            x, y = a[key], b[key]
            if x.shape != y.shape:
                problems.append(f"{key}: shape {x.shape} != {y.shape}")
                continue
            if x.dtype != y.dtype:
                problems.append(f"{key}: dtype {x.dtype} != {y.dtype}")
            if np.issubdtype(x.dtype, np.floating) or np.issubdtype(y.dtype, np.floating):
                xf, yf = x.astype(np.float64), y.astype(np.float64)
                same_nan = np.array_equal(np.isnan(xf), np.isnan(yf))
                diff = np.abs(np.where(np.isnan(xf), 0, xf) - np.where(np.isnan(yf), 0, yf))
                worst = float(diff.max()) if diff.size else 0.0
                if not same_nan or worst > atol:
                    problems.append(f"{key}: max abs difference {worst:.3g}"
                                    + ("" if same_nan else ", NaN pattern differs"))
            elif not np.array_equal(x, y):
                n = int(np.sum(x != y))
                problems.append(f"{key}: {n} of {x.size} entries differ")
    return problems


def pairs(new_root: Path, ref_root: Path | None):
    if ref_root is not None:
        for new in sorted(new_root.rglob("*.npz")):
            yield new, ref_root / new.relative_to(new_root)
        return
    for new in sorted(new_root.rglob("*.npz")):
        ref = DATA / new.relative_to(new_root)
        if ref.exists():
            yield new, ref


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("regenerated", help="regenerated output root or cache directory")
    ap.add_argument("reference", nargs="?", help="reference directory (default: matching paths under data/)")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance for floating-point arrays")
    ap.add_argument("--quiet", action="store_true", help="print only mismatches and the summary")
    args = ap.parse_args(argv)
    new_root = Path(args.regenerated).resolve()
    ref_root = Path(args.reference).resolve() if args.reference else None
    compared = failed = missing = 0
    for new, ref in pairs(new_root, ref_root):
        if not ref.exists():
            print(f"NO REFERENCE {new}")
            missing += 1
            continue
        compared += 1
        problems = compare_file(new, ref, args.atol)
        if problems:
            failed += 1
            print(f"MISMATCH {new.relative_to(new_root)}")
            for p in problems:
                print("   ", p)
        elif not args.quiet:
            print(f"identical {new.relative_to(new_root)}")
    print(f"compared {compared} file(s): {compared - failed} match, {failed} differ"
          + (f", {missing} without reference" if missing else ""))
    return 0 if compared and not failed and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
