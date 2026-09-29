#!/usr/bin/env python3
"""Replay the 30-group FEMNIST training deterministically from round 80 to 120 and score the intermediate states.

Loads the round-80 checkpoint written by train.py, repeats the same federated
averaging rounds (same client samples and minibatch orders), saves the models
at rounds 81, 82, 84, 88, 96, 104, 112 and 120, and scores every round in
80, 81, ..., 120 with the confirmation seeds 2-5 and one candidate epoch.
Output names match data/femnist30/decay/. The replayed round-120 model is
checked against the one saved by train.py.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, output_dir  # noqa: E402

R_INT = (81, 82, 84, 88, 96, 104, 112)
ROUNDS = (80, *R_INT, 120)
SEEDS = (2, 3, 4, 5)


def replay(out: Path, group: int, parts_cache) -> Path:
    import torch
    from model import fedavg_round, new_model
    from train import CLIENTS_PER_ROUND, GROUP_TAG, INIT_SEED, checkpoint_dir

    writers, _, parts = parts_cache
    source = checkpoint_dir(out, group)
    target = out / "femnist30" / "decay-checkpoints" / f"group-{group:02d}"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "round-80.pt", target / "round-80.pt")
    model = new_model(INIT_SEED)
    model.load_state_dict(torch.load(source / "round-80.pt", weights_only=True))
    for r in range(80, 121):
        if r in ROUNDS and r != 80:
            torch.save(model.state_dict(), target / f"round-{r}.pt")
        if r < 120:
            fedavg_round(model, parts, writers, r, GROUP_TAG.format(group=group),
                         clients_per_round=CLIENTS_PER_ROUND)
    reference = source / "round-120.pt"
    if reference.exists():
        ref = torch.load(reference, weights_only=True)
        mine = torch.load(target / "round-120.pt", weights_only=True)
        diff = max(float((ref[k] - mine[k]).abs().max()) for k in ref)
        print("group", group, "replayed round-120 max abs difference from train.py:", diff, flush=True)
    return target


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(34)))
    ap.add_argument("--rounds", type=int, nargs="+", default=list(ROUNDS), choices=ROUNDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    from prepare import load_group
    from score import score_state
    for group in args.groups:
        parts = load_group(out, group)
        started = time.monotonic()
        checkpoints = replay(out, group, parts)
        for r in args.rounds:
            for s in args.seeds:
                path = score_state(out, group, r, s, "confirmation", 1, checkpoints=checkpoints,
                                   cache_dir=out / "femnist30" / "decay", parts_cache=parts)
                print("scored", path.name, "elapsed", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
