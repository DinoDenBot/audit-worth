#!/usr/bin/env python3
"""Train the federated text MLP of each Stack Overflow group for 80 rounds."""
import argparse
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, output_dir  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(1, 9)), choices=range(9))
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    from model import train_group
    for group in args.groups:
        train_group(out, group)


if __name__ == "__main__":
    main()
