#!/usr/bin/env python3
"""Score 30-group FEMNIST federation states: per-record correctness of every action and exact Shapley values.

For a saved round-r model and a seed, each of the four contributors forms a
candidate update (``epochs`` Adam epochs on its training role). On each
provider's complete working inventory the script records the 0/1 correctness
of all 16 coalition models (mean of the chosen contributors' updates) and
stores keep-current plus the four single-contributor actions (``bits_p``, 5 x n)
and every contributor's exact four-player per-record Shapley value
(``psi_player{j}_provider{p}``). Output names match data/femnist30/state/.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, output_dir  # noqa: E402

CHECKPOINTS = (80, 120, 160)
DEV_SEEDS = (0, 1)
CONF_SEEDS = (2, 3, 4, 5)
PLAYER_COUNT = 4
LOCAL_LR = .001
INIT_SEED = 270927
# Fixed label mixed into the candidate-update random streams.
CANDIDATE_TAG = "decision-map:candidate:{group}:{round}:{seed}:{i}:{writer}"


def target(parts, writers, role):
    """Concatenate one role over all writers with equal-writer weights."""
    blocks = [parts[w][role] for w in writers]
    x = np.concatenate([b[0] for b in blocks])
    y = np.concatenate([b[1] for b in blocks])
    weights = np.concatenate([np.full(len(b[1]), 1 / (len(writers) * len(b[1])), dtype=np.float64)
                              for b in blocks])
    return (x, y, None), weights


def shapley(bits: np.ndarray, player: int) -> np.ndarray:
    """Exact four-player Shapley value per record from 16 coalition correctness rows."""
    result = np.zeros(bits.shape[1], dtype=np.float64)
    for mask in range(16):
        if mask & (1 << player):
            continue
        size = mask.bit_count()
        weight = 1 / (4 * math.comb(3, size))
        result += weight * (bits[mask | (1 << player)].astype(float) - bits[mask].astype(float))
    return result


def score_state(out: Path, group: int, round_index: int, seed: int, phase: str, epochs: int = 1,
                checkpoints: Path | None = None, cache_dir: Path | None = None, parts_cache=None) -> Path:
    import torch
    from model import coalition_model, local_fit, new_model, predict
    from prepare import load_group
    from train import checkpoint_dir

    if phase not in ("development", "confirmation"):
        raise ValueError(phase)
    if epochs not in (1, 3):
        raise ValueError(epochs)
    cache_dir = cache_dir or out / "femnist30" / "state"
    checkpoints = checkpoints or checkpoint_dir(out, group)
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if epochs == 1 else f"-e{epochs}"
    path = cache_dir / f"g{group:02d}-r{round_index}-s{seed}{suffix}-{phase}.npz"
    if path.exists():
        return path
    writers, players, parts = parts_cache or load_group(out, group)
    model = new_model(INIT_SEED)
    model.load_state_dict(torch.load(checkpoints / f"round-{round_index}.pt", weights_only=True))
    candidates = [local_fit(copy.deepcopy(model), parts[writer][0],
                            CANDIDATE_TAG.format(group=group, round=round_index, seed=seed, i=i, writer=writer),
                            epochs=epochs, lr=LOCAL_LR) for i, writer in enumerate(players)]
    actions = [model, *candidates]
    dev, dev_weights = target(parts, writers, 2)
    dev_acc = np.asarray([float((predict(action, dev) == dev[1]) @ dev_weights) for action in actions])
    if phase == "confirmation":
        inv, inv_weights = target(parts, writers, 1)
        ext, ext_weights = target(parts, writers, 3)
        target_acc = np.asarray([float((predict(action, inv) == inv[1]) @ inv_weights) for action in actions])
        external_acc = np.asarray([float((predict(action, ext) == ext[1]) @ ext_weights) for action in actions])
    else:
        target_acc = np.full(5, np.nan)
        external_acc = np.full(5, np.nan)
    blocks = [parts[w][1] for w in players]
    spans = np.cumsum([0] + [len(block[1]) for block in blocks])
    own = (np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]), None)
    correct = np.stack([(predict(coalition_model(model, candidates, mask), own) == own[1]).astype(np.uint8)
                        for mask in range(16)])
    payload: dict[str, np.ndarray] = {"dev_acc": dev_acc, "target_acc": target_acc, "external_acc": external_acc}
    for i, block in enumerate(blocks):
        bits = correct[:, spans[i]:spans[i + 1]]
        payload[f"bits_{i}"] = bits[[0, 1, 2, 4, 8]]
        payload[f"psi_{i}"] = shapley(bits, i)
        payload[f"proxy_{i}"] = bits[1 << i].astype(float) - bits[0].astype(float)
        payload[f"rows_{i}"] = block[2]
        for player in range(PLAYER_COUNT):
            payload[f"psi_player{player}_provider{i}"] = shapley(bits, player)
    np.savez_compressed(path, **payload)
    return path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(34)))
    ap.add_argument("--rounds", type=int, nargs="+", default=list(CHECKPOINTS), choices=CHECKPOINTS)
    ap.add_argument("--phase", choices=("development", "confirmation", "both"), default="both",
                    help="development uses seeds 0-1 (no target accuracies), confirmation seeds 2-5")
    ap.add_argument("--seeds", type=int, nargs="+", help="restrict to these seeds")
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 3], choices=(1, 3))
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    from prepare import load_group
    phases = ("development", "confirmation") if args.phase == "both" else (args.phase,)
    for group in args.groups:
        parts = load_group(out, group)
        started = time.monotonic()
        for epochs in args.epochs:
            for phase in phases:
                seeds = DEV_SEEDS if phase == "development" else CONF_SEEDS
                for round_index in args.rounds:
                    for seed in seeds:
                        if args.seeds and seed not in args.seeds:
                            continue
                        path = score_state(out, group, round_index, seed, phase, epochs, parts_cache=parts)
                        print("scored", path.name, "elapsed", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
