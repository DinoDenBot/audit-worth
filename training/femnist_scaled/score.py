#!/usr/bin/env python3
"""Score update-scaling FEMNIST states: candidate updates scaled by 0.5, 1 or 2 and per-record correctness/Shapley values.

For each group, confirmation seed (4-11) and state (the contributor whose
update branches the round-160 model), each contributor forms a candidate
update; the update is multiplied by the scale factor. The script records the
candidates' correctness on the pooled working inventory (``target_correct``)
and their equal-writer accuracies, and, on each provider's working inventory,
the four candidates' 0/1 correctness (``candidate_bits_p``) and the provider's
exact four-player per-record Shapley value (``psi_p``). Output names match
data/femnist_scaled/state/.
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

SCALES = (0.5, 1.0, 2.0)
CONF_SEEDS = tuple(range(4, 12))
INIT_SEED = 270927


def concat(parts, writers, role):
    blocks = [parts[w][role] for w in writers]
    weights = np.concatenate([np.full(len(b[1]), 1 / (len(writers) * len(b[1]))) for b in blocks])
    return (np.concatenate([b[0] for b in blocks]), np.concatenate([b[1] for b in blocks]), None), weights


def scale_candidates(state, candidates, factor):
    import torch
    baseline = state.state_dict()
    scaled = []
    for candidate in candidates:
        model = copy.deepcopy(state)
        weights = candidate.state_dict()
        with torch.no_grad():
            model.load_state_dict({key: baseline[key] + factor * (weights[key] - baseline[key]) for key in baseline})
        scaled.append(model)
    return scaled


def credit(bits, player):
    """Exact four-player Shapley value per record."""
    out = np.zeros(bits.shape[1], dtype=float)
    for mask in range(16):
        if mask & (1 << player):
            continue
        size = mask.bit_count()
        weight = 1 / (4 * math.comb(3, size))
        out += weight * (bits[mask | (1 << player)].astype(float) - bits[mask].astype(float))
    return out


def state_cache(checkpoint, parts, writers, players, seed, state_index, factor, dev_target, target, external):
    from model import coalition_model, predict, terminal_models

    branch, original = terminal_models(checkpoint, parts, players, seed, state_index)
    candidates = scale_candidates(branch, original, factor)
    target_correct = np.stack([(predict(model, target[0]) == target[0][1]).astype(np.uint8) for model in candidates])
    payload = {
        "target_correct": target_correct,
        "target_acc": target_correct @ target[1],
        "dev_acc": np.asarray([(predict(model, dev_target[0]) == dev_target[0][1]) @ dev_target[1]
                               for model in candidates]),
        "external_acc": np.asarray([(predict(model, external[0]) == external[0][1]) @ external[1]
                                    for model in candidates]),
    }
    own = [parts[w][1] for w in players]
    spans = np.cumsum([0] + [len(b[1]) for b in own])
    combined = (np.concatenate([b[0] for b in own]), np.concatenate([b[1] for b in own]), None)
    coalition_bits = np.stack([(predict(coalition_model(branch, candidates, mask), combined)
                                == combined[1]).astype(np.uint8) for mask in range(16)])
    for i, writer in enumerate(players):
        bits = coalition_bits[:, spans[i]:spans[i + 1]]
        payload[f"candidate_bits_{i}"] = bits[[1 << j for j in range(4)]]
        payload[f"psi_{i}"] = credit(bits, i)
        payload[f"source_rows_{i}"] = parts[writer][1][2]
    return payload


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4], choices=(1, 2, 3, 4))
    ap.add_argument("--scales", type=float, nargs="+", default=list(SCALES), choices=SCALES)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(CONF_SEEDS), choices=CONF_SEEDS)
    ap.add_argument("--states", type=int, nargs="+", default=[0, 1, 2, 3], choices=range(4))
    args = ap.parse_args(argv)
    out = output_dir(args.out)
    import torch
    from model import new_model
    from prepare import load_group
    from train import checkpoint_dir

    cache_dir = out / "femnist_scaled" / "state"
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for group in args.groups:
        writers, players, parts = load_group(out, group)
        checkpoint = new_model(INIT_SEED)
        checkpoint.load_state_dict(torch.load(checkpoint_dir(out, group) / "round-160.pt", weights_only=True))
        dev, target, external = (concat(parts, writers, role) for role in (2, 1, 3))
        for factor in args.scales:
            for seed in args.seeds:
                for state_index in args.states:
                    path = cache_dir / f"g{group}-x{factor:g}-z{seed}-a{state_index}.npz"
                    if path.exists():
                        continue
                    payload = state_cache(checkpoint, parts, writers, players, seed, state_index, factor,
                                          dev, target, external)
                    np.savez_compressed(path, **payload)
                    print("scored", path.name, "elapsed", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
