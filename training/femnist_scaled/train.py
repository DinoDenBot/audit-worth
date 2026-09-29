#!/usr/bin/env python3
"""Train the deterministic federated-averaging trajectory of one 64-writer FEMNIST group (update-scaling study).

Sixteen of the 64 writers are sampled per round; checkpoints (including the
round-160 model that all states branch from) are saved under
<out>/femnist_scaled/checkpoints/group-G/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, output_dir  # noqa: E402

CHECKPOINTS = (0, 5, 10, 20, 40, 60, 80, 100, 120, 160)
INIT_SEED = 270927


def checkpoint_dir(out: Path, group: int) -> Path:
    return out / "femnist_scaled" / "checkpoints" / f"group-{group}"


def train(out: Path, group: int, rounds: int = 160) -> None:
    import torch
    from model import equal_user_accuracy, fedavg_round, new_model
    from prepare import load_group

    writers, _, parts = load_group(out, group)
    model = new_model(INIT_SEED)
    target = checkpoint_dir(out, group)
    target.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    for r in range(rounds + 1):
        if r in CHECKPOINTS:
            torch.save(model.state_dict(), target / f"round-{r}.pt")
            dev = equal_user_accuracy(model, parts, writers, 2)
            records.append({"round": r, "development_accuracy": dev, "elapsed_seconds": time.monotonic() - started})
            print("group", group, "round", r, "dev", round(dev, 4),
                  "elapsed", round(records[-1]["elapsed_seconds"], 1), flush=True)
        if r < rounds:
            fedavg_round(model, parts, writers, r, str(group))
    (target / "quality.json").write_text(json.dumps({"group": group, "checkpoints": records}, indent=2) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4], choices=range(5))
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    for group in args.groups:
        train(out, group)


if __name__ == "__main__":
    main()
