#!/usr/bin/env python3
"""Run every reproduction script and summarize the comparison with the paper.

Usage:
    python reproduce.py            # all scripts
    python reproduce.py --list     # list scripts and what they reproduce
    python reproduce.py table2 fig4   # scripts whose names contain these substrings
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def scripts():
    return sorted(p for p in (ROOT / "scripts").glob("*.py") if not p.name.startswith("_"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("only", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    chosen = [p for p in scripts() if not a.only or any(s in p.stem for s in a.only)]
    if a.list:
        for p in chosen:
            doc = (p.read_text().split('"""')[1].strip().splitlines() or [""])[0]
            print(f"{p.stem:40s} {doc}")
        return
    failed = []
    for p in chosen:
        t0 = time.time()
        spec = importlib.util.spec_from_file_location(p.stem, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok = mod.main()
        print(f"== {p.stem}: {'PASS' if ok else 'FAIL'} ({time.time() - t0:.0f}s)\n")
        if not ok:
            failed.append(p.stem)
    print("All checks passed." if not failed else f"Failed: {', '.join(failed)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
