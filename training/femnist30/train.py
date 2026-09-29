#!/usr/bin/env python3
"""Train the deterministic federated-averaging trajectory of each 30-group FEMNIST federation.

Four of the eight writers are sampled per round; checkpoints are saved at rounds
0, 20, 40, 80, 120 and 160 under <out>/femnist30/checkpoints/group-XX/.
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

CHECKPOINTS = (0, 20, 40, 80, 120, 160)
INIT_SEED = 270927
CLIENTS_PER_ROUND = 4
# Fixed label mixed into the client-sampling and minibatch random streams.
GROUP_TAG = "decision-map:{group}"


def checkpoint_dir(out: Path, group: int) -> Path:
    return out / "femnist30" / "checkpoints" / f"group-{group:02d}"


def train(out: Path, group: int, limit: int = 160) -> None:
    import torch
    from model import equal_user_accuracy, fedavg_round, new_model
    from prepare import load_group

    writers, _, parts = load_group(out, group)
    target = checkpoint_dir(out, group)
    target.mkdir(parents=True, exist_ok=True)
    model = new_model(INIT_SEED)
    rows = []
    started = time.monotonic()
    for round_index in range(limit + 1):
        if round_index in CHECKPOINTS:
            torch.save(model.state_dict(), target / f"round-{round_index}.pt")
            row = {"round": round_index,
                   "development_accuracy": equal_user_accuracy(model, parts, writers, 2),
                   "training_accuracy": equal_user_accuracy(model, parts, writers, 0),
                   "elapsed_seconds": time.monotonic() - started}
            rows.append(row)
            print("group", group, "round", round_index, "dev", round(row["development_accuracy"], 4),
                  "elapsed", round(row["elapsed_seconds"], 1), flush=True)
        if round_index < limit:
            fedavg_round(model, parts, writers, round_index, GROUP_TAG.format(group=group),
                         clients_per_round=CLIENTS_PER_ROUND)
    (target / "quality.json").write_text(json.dumps({"group": group, "checkpoints": rows}, indent=2) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(34)))
    ap.add_argument("--rounds", type=int, default=160, choices=CHECKPOINTS[1:])
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    for group in args.groups:
        train(out, group, args.rounds)


if __name__ == "__main__":
    main()
