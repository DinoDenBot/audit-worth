#!/usr/bin/env python3
"""Regenerate the UCI HAR federation states: a 30-subject federation with a multinomial logistic classifier.

Each subject is a client. Every subject's windows (train and test files
combined, in file order) are split into roles 0 = local training (first half),
1 = working inventory, 2 = development target, 3 = external target, with one
guard window between consecutive roles. Four subjects are the contributors and
evaluation providers. For each seed the global model is trained by two epochs
of SGD on all subjects' training roles; for each state one contributor's local
update branches it and each contributor then forms a candidate update. On each
provider's working inventory the script records the four candidates' 0/1
correctness (``bits_p``) and the provider's exact four-player per-record
Shapley value (``psi_p``). Output names match data/har/state/.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import math
import os
from pathlib import Path
import sys
import time
import zipfile

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, DOWNLOADS, check_source, output_dir  # noqa: E402

SOURCE_SHA = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
DEFAULT_SOURCE = DOWNLOADS / "uci-har.zip"
SEEDS = tuple(range(8, 28))
CLASSES = np.arange(6)
SHAPLEY_TERMS = tuple(
    (i, mask, mask | (1 << i), math.factorial(mask.bit_count()) * math.factorial(3 - mask.bit_count()) / 24)
    for i in range(4) for mask in range(16) if not mask & (1 << i)
)


# The string prefixes below are fixed domain-separation labels for the random
# streams and the contributor choice; changing them changes every result.
def hkey_int(prefix, s):
    return int(hashlib.sha256((prefix + str(s)).encode()).hexdigest(), 16)


def seed_for(prefix: str, value: str) -> int:
    return int.from_bytes(hashlib.sha256((prefix + value).encode()).digest()[:8], "big")


def parts_from_blocks(blocks):
    assert all(len(b) > 0 for b in blocks)
    return tuple((np.stack([r[1] for r in b]).astype(np.float32), np.array([r[2] for r in b], dtype=np.int64),
                  [r[0] for r in b]) for b in blocks)


def har_data(source: Path):
    """Return (subjects, contributors, parts) from the UCI download (outer zip containing 'UCI HAR Dataset.zip')."""
    with zipfile.ZipFile(source) as outer:
        payload = outer.read("UCI HAR Dataset.zip")
    by = {str(i): [] for i in range(1, 31)}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for split in ("train", "test"):
            root = f"UCI HAR Dataset/{split}/"
            x = np.loadtxt(io.BytesIO(z.read(root + f"X_{split}.txt")), dtype=np.float32)
            y = np.loadtxt(io.BytesIO(z.read(root + f"y_{split}.txt")), dtype=np.int64) - 1
            ids = np.loadtxt(io.BytesIO(z.read(root + f"subject_{split}.txt")), dtype=np.int64)
            assert x.shape[1] == 561 and len(x) == len(y) == len(ids)
            np.clip(x, -1, 1, out=x)
            for j, (id_, feat, label) in enumerate(zip(ids, x, y)):
                by[str(id_)].append((j, feat, int(label)))
    parts = {}
    for w, rs in by.items():
        n = len(rs)
        a = n // 2
        b = a + 1
        c = b + n // 4
        d = c + 1
        e = d + n // 8
        f = e + 1
        parts[w] = parts_from_blocks([rs[:a], rs[b:c], rs[d:e], rs[f:]])
    writers = sorted(by, key=int)
    players = sorted(writers, key=lambda w: hkey_int("cross-curation-contributor-v1:", w))[:4]
    assert min(len(parts[w][1][1]) for w in players) >= 16
    return writers, players, parts


def concat(parts, writers, slot):
    xs, ys, weights, spans = [], [], [], {}
    start = 0
    for w in writers:
        x, y, _ = parts[w][slot]
        xs.append(x)
        ys.append(y)
        weights.append(np.full(len(y), 1 / (len(writers) * len(y)), dtype=np.float64))
        spans[w] = slice(start, start + len(y))
        start += len(y)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights), spans


def local_update(model, x, y, seed, tag):
    """One pass of the SGD classifier over a client's training role in a hash-seeded order."""
    order = np.random.default_rng(seed_for("curation-local-v1:", f"{seed}:{tag}")).permutation(len(y))
    model.partial_fit(x[order], y[order])
    return model


def train_global(parts, writers, seed):
    from sklearn.linear_model import SGDClassifier

    x = np.concatenate([parts[w][0][0] for w in writers])
    y = np.concatenate([parts[w][0][1] for w in writers])
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, learning_rate="constant", eta0=0.02,
                          average=False, shuffle=False, random_state=seed)
    for epoch in range(2):
        order = np.random.default_rng(seed_for("curation-global-v1:", f"{seed}:{epoch}")).permutation(len(y))
        if epoch == 0:
            model.partial_fit(x[order], y[order], classes=CLASSES)
        else:
            model.partial_fit(x[order], y[order])
    return model


def coalition_models(state, candidates):
    """Model for every coalition mask: the state plus the mean of the chosen contributors' updates."""
    models = [state]
    for mask in range(1, 16):
        chosen = [j for j in range(4) if mask & (1 << j)]
        model = copy.deepcopy(state)
        model.coef_ = state.coef_ + np.mean([candidates[j].coef_ - state.coef_ for j in chosen], axis=0)
        model.intercept_ = state.intercept_ + np.mean([candidates[j].intercept_ - state.intercept_ for j in chosen],
                                                      axis=0)
        models.append(model)
    return models


def score_bits(models, x, y):
    return np.stack([(model.predict(x) == y).astype(np.uint8) for model in models])


def shapley_per_record(bits):
    psi = np.zeros((4, bits.shape[1]), dtype=np.float64)
    for i, lo, hi, weight in SHAPLEY_TERMS:
        psi[i] += weight * (bits[hi].astype(float) - bits[lo].astype(float))
    return psi


def seed_states(seed, writers, players, parts, target_inv):
    """The four states of one seed; per provider (candidate bits (4, n), own Shapley values (n,))."""
    base = train_global(parts, writers, seed)
    x, y, _, spans = target_inv
    states = []
    for a, branch in enumerate(players):
        state = local_update(copy.deepcopy(base), *parts[branch][0][:2], seed, f"branch:{a}:{branch}")
        candidates = [local_update(copy.deepcopy(state), *parts[w][0][:2], seed, f"second:{a}:{j}:{w}")
                      for j, w in enumerate(players)]
        all_bits = score_bits(coalition_models(state, candidates), x, y)
        providers = []
        for i, w in enumerate(players):
            b = all_bits[:, spans[w]]
            providers.append((b[[1 << j for j in range(4)]], shapley_per_record(b)[i]))
        states.append(providers)
    return states


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="UCI HAR download (zip)")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    ap.add_argument("--skip-digest", action="store_true")
    args = ap.parse_args(argv)
    source = check_source(args.source, SOURCE_SHA, args.skip_digest)
    target = output_dir(args.out) / "har" / "state"
    target.mkdir(parents=True, exist_ok=True)
    writers, players, parts = har_data(source)
    target_inv = concat(parts, writers, 1)
    started = time.monotonic()
    for seed in args.seeds:
        for state, providers in enumerate(seed_states(seed, writers, players, parts, target_inv)):
            payload = {}
            for p, (bits, psi) in enumerate(providers):
                payload[f"bits_{p}"] = bits.astype(np.uint8)
                payload[f"psi_{p}"] = np.asarray(psi, dtype=np.float64)
            np.savez_compressed(target / f"s{seed:02d}-a{state}.npz", **payload)
        print("seed", seed, "done", "elapsed", round(time.monotonic() - started, 1), flush=True)


if __name__ == "__main__":
    main()
