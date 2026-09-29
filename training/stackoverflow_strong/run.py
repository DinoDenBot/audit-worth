#!/usr/bin/env python3
"""Regenerate the stronger Stack Overflow states: 120-round federated sparse character-feature logistic model.

Uses the same users, roles and labels as the Stack Overflow text study
(../stackoverflow/client-manifest.json), with 8192-dimensional hashed
character 3-5-gram features read directly from the FedJAX SQLite file. The
model is trained by 120 rounds of federated averaging (8 of 16 users per round,
one SGD pass each). For each analysed seed (4-19) and state, one contributor's
update branches the round-120 model and each contributor forms a candidate
update. For each provider's working inventory the script stores the
candidates' 0/1 correctness (``bits_p``, 4 x n) and the provider's exact
four-player per-record Shapley value (``psi_p``), plus the candidates'
equal-user inventory accuracy (``full_accuracy``) and their mean accuracy over
the four providers (``provider_accuracy``). Output names match
data/stackoverflow_strong/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "stackoverflow"))
from common import DEFAULT_OUT, output_dir, check_source  # noqa: E402
from prepare import DEFAULT_SOURCE, MANIFEST, SOURCE_SHA, read_user, user_blocks  # noqa: E402

ETA = .2
ALPHA = 1e-5
ROUNDS = 120
SEEDS = tuple(range(4, 20))
FEATURES = 8192
# Fixed integer seed and string labels of the random streams; changing them changes every result.
SAMPLE_SEED = 290929


def intseed(*parts) -> int:
    return int.from_bytes(hashlib.sha256(":".join(map(str, parts)).encode()).digest()[:8], "big") % (2**32)


def load(source: Path, group: int):
    """Return (users, players, data); data[user][role] = (sparse features, labels)."""
    from sklearn.feature_extraction.text import HashingVectorizer

    manifest = json.loads(MANIFEST.read_text())
    uids = manifest["groups"][group]
    conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    vec = HashingVectorizer(n_features=FEATURES, analyzer="char_wb", ngram_range=(3, 5),
                            alternate_sign=False, norm="l2")
    data = {}
    for uid in uids:
        blocks = user_blocks(uid, read_user(conn, uid))
        data[uid] = [(vec.transform([r[1] for r in block]).tocsr(), np.asarray([r[2] for r in block]))
                     for block in blocks]
    conn.close()
    return uids, manifest["players"][group], data


def fresh_model(eta: float, alpha: float, seed: int, weights=None):
    """A binary logistic SGDClassifier initialised at zero (or at ``weights``)."""
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import SGDClassifier

    cls = SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha, learning_rate="constant", eta0=eta,
                        shuffle=True, random_state=seed, average=False)
    cls.partial_fit(csr_matrix((1, FEATURES)), np.array([0]), classes=np.array([0, 1]))
    cls.coef_[:] = 0
    cls.intercept_[:] = 0
    cls.t_ = 1.0
    if weights is not None:
        cls.coef_[:] = weights[0]
        cls.intercept_[:] = weights[1]
    return cls


def update(weights, block, eta: float, alpha: float, seed: int):
    """One SGD pass over a client's training role (shuffled with ``seed``)."""
    cls = fresh_model(eta, alpha, seed, weights)
    cls.partial_fit(*block)
    return cls.coef_.copy(), cls.intercept_.copy()


def accuracy(weights, data, uids, role=1):
    model = fresh_model(.1, 1e-5, 0, weights)
    return float(np.mean([(model.predict(data[uid][role][0]) == data[uid][role][1]).mean() for uid in uids]))


def train(uids, data, eta, alpha, rounds=ROUNDS):
    weights = (np.zeros((1, FEATURES)), np.zeros(1))
    history = []
    for rnd in range(1, rounds + 1):
        chosen = np.random.default_rng([SAMPLE_SEED, rnd]).choice(len(uids), 8, replace=False)
        updates = [update(weights, data[uids[i]][0], eta, alpha, SAMPLE_SEED + rnd * 100 + i) for i in chosen]
        weights = (np.mean([x[0] for x in updates], axis=0), np.mean([x[1] for x in updates], axis=0))
        if rnd in (1, 20, 40, 60, 80, 100, 120):
            acc = accuracy(weights, data, uids)
            history.append((rnd, acc))
            print("round", rnd, f"inventory accuracy {acc:.6f}", flush=True)
    return weights, history


def prediction(weights, X):
    return (np.asarray(X @ weights[0].T).ravel() + weights[1][0] > 0).astype(np.uint8)


def coalition(state, updates, mask):
    """Coalition model: the average of the chosen contributors' updated weights (mask 0 keeps the state)."""
    if mask == 0:
        return state
    chosen = [updates[j] for j in range(4) if mask & (1 << j)]
    return (np.mean([x[0] for x in chosen], axis=0), np.mean([x[1] for x in chosen], axis=0))


def shapley(bits, player):
    value = np.zeros(bits.shape[1])
    for mask in range(16):
        if mask & (1 << player):
            continue
        size = mask.bit_count()
        weight = 1 / (4 * math.comb(3, size))
        value += weight * (bits[mask | (1 << player)].astype(float) - bits[mask].astype(float))
    return value


def run_group(source: Path, out: Path, group: int, seeds, states) -> None:
    target = out / "stackoverflow_strong" / f"group-{group}"
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    uids, players, data = load(source, group)
    weights, history = train(uids, data, ETA, ALPHA)
    model_dir = out / "stackoverflow_strong" / "checkpoints" / f"group-{group}"
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_dir / "round-120.npz", coef=weights[0], intercept=weights[1])
    (model_dir / "quality.json").write_text(json.dumps({"group": group, "history": history}, indent=2) + "\n")
    for seed in seeds:
        for state_idx in states:
            state = update(weights, data[players[state_idx]][0], ETA, ALPHA,
                           intseed("strong-state", group, seed, state_idx))
            updates = [update(state, data[uid][0], ETA, ALPHA, intseed("strong-terminal", group, seed, state_idx, j))
                       for j, uid in enumerate(players)]
            masks = [coalition(state, updates, mask) for mask in range(16)]
            payload = {"group": group, "seed": seed, "state": state_idx}
            provider_bits = []
            for p, uid in enumerate(players):
                X, y = data[uid][1]
                bits16 = np.stack([prediction(w, X) == y for w in masks]).astype(np.uint8)
                bits = bits16[[1, 2, 4, 8]]
                provider_bits.append(bits)
                payload[f"bits_{p}"] = bits
                payload[f"psi_{p}"] = shapley(bits16, p)
            payload["full_accuracy"] = np.mean([
                [(prediction(updates[j], data[uid][1][0]) == data[uid][1][1]).mean() for j in range(4)]
                for uid in uids], axis=0)
            payload["provider_accuracy"] = np.mean([bits.mean(axis=1) for bits in provider_bits], axis=0)
            np.savez_compressed(target / f"score-s{seed:02d}-a{state_idx}.npz", **payload)
        print("group", group, "seed", seed, "scored", "elapsed", round(time.monotonic() - started, 1), flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="FedJAX stackoverflow_train.sqlite")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(1, 9)), choices=range(1, 9))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), choices=SEEDS)
    ap.add_argument("--states", type=int, nargs="+", default=[0, 1, 2, 3], choices=range(4))
    ap.add_argument("--skip-digest", action="store_true")
    args = ap.parse_args(argv)
    source = check_source(args.source, SOURCE_SHA, args.skip_digest)
    out = output_dir(args.out)
    for group in args.groups:
        run_group(source, out, group, args.seeds, args.states)


if __name__ == "__main__":
    main()
