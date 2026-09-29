#!/usr/bin/env python3
"""Score Stack Overflow states: natural local-update branches and exact four-player per-record Shapley values.

For each group, seed (0-3 design, 4-19 analysed) and state, one contributor's
local update branches the round-80 model and each contributor forms a candidate
update from the branched model. The script stores the candidates' equal-user
accuracies on the target, development and external roles, and for each
provider's working inventory the candidates' 0/1 correctness (``bits_p``,
4 x n), all four contributors' per-record Shapley values (``psi_p``, 4 x n) and
the source indices of the records. Output names match data/stackoverflow/.
"""
import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, output_dir  # noqa: E402


def shapley(correct):
    """correct[mask, record] is coalition correctness (mask 0 = keep current)."""
    out = np.zeros((4, correct.shape[1]), dtype=np.float64)
    for player in range(4):
        for mask in range(16):
            if mask & (1 << player):
                continue
            n = mask.bit_count()
            weight = math.factorial(n) * math.factorial(3 - n) / math.factorial(4)
            out[player] += weight * (correct[mask | (1 << player)].astype(float) - correct[mask].astype(float))
    return out


def score_state(group, seed, state, data, checkpoint):
    from model import coalition_params, concat_role, intseed, local_update, predict, target_accuracy

    writers, players, data = data
    current = local_update(checkpoint, data[players[state]][0], intseed("state-v1", group, seed, state), epochs=1)
    updates = [local_update(current, data[uid][0], intseed("terminal-v1", group, seed, state, uid), epochs=1)
               for uid in players]
    out = {"group": group, "seed": seed, "state": state}
    for role, name in ((1, "target"), (2, "development"), (3, "external")):
        x, y, slices = concat_role(writers, data, role)
        out[f"{name}_accuracy"] = np.array([target_accuracy(predict(update, x), y, slices) for update in updates],
                                           dtype=np.float64)
    for provider, uid in enumerate(players):
        x, y, row_ids = data[uid][1]
        y = y.numpy()
        bits = np.stack([predict(coalition_params(current, updates, mask), x) == y for mask in range(16)]).astype(np.uint8)
        out[f"bits_{provider}"] = bits[[1, 2, 4, 8]]
        out[f"psi_{provider}"] = shapley(bits)
        out[f"row_ids_{provider}"] = row_ids
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(1, 9)), choices=range(9))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)), choices=range(20))
    ap.add_argument("--states", type=int, nargs="+", default=[0, 1, 2, 3], choices=range(4))
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    import torch
    from model import load_group
    started = time.monotonic()
    for group in args.groups:
        data = load_group(out, group)
        checkpoint = torch.load(out / "stackoverflow" / "checkpoints" / f"group-{group}" / "round-80.pt",
                                map_location="cpu", weights_only=True)
        target = out / "stackoverflow" / f"group-{group}"
        target.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            for state in args.states:
                path = target / f"score-s{seed:02d}-a{state}.npz"
                if path.exists():
                    continue
                np.savez_compressed(path, **score_state(group, seed, state, data, checkpoint))
            print("group", group, "seed", seed, "scored", "elapsed", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
