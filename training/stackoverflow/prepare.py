#!/usr/bin/env python3
"""Extract the Stack Overflow users of the federated text study from the FedJAX SQLite file and save hashed features.

A record is one post; the binary label is its ``type`` (question or answer) and
the only input is its ``tokens`` text. Each post is assigned to a role by a
fixed hash of (user, creation date, title, type): 0 = local training (1/2),
1 = working inventory (1/4), 2 = development target (1/8), 3 = external target.
Features are 1024-dimensional hashed word 1-2-grams. The 144 users (group 0
for design, groups 1-8 analysed, 16 users each) are fixed in client-manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import zlib

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, DOWNLOADS, check_source, output_dir  # noqa: E402

MANIFEST = HERE / "client-manifest.json"
SOURCE_SHA = "3efde71d939fd7cb58d357dd3c060c3a8c8babec7607df9c9cc549034bafc646"
DEFAULT_SOURCE = DOWNLOADS / "stackoverflow_train.sqlite"
# Fixed domain-separation string for the post-to-role assignment; changing it changes every split.
ROLE_PREFIX = b"so-trained-natural-role-v1:"
FEATURES = 1024


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def prepared_dir(out: Path) -> Path:
    return out / "stackoverflow" / "prepared"


def ext(code, data):
    """msgpack extension hook for the NumPy arrays stored in the FedJAX SQLite file."""
    import msgpack
    if code == 4:
        shape, flat = msgpack.unpackb(data, raw=True)
        return np.array(flat, dtype=object).reshape(shape)
    if code in (1, 3):
        shape, dtype, buf = msgpack.unpackb(data, raw=True)
        arr = np.frombuffer(buf, dtype=np.dtype(dtype)).reshape(shape)
        return arr[()] if code == 3 else arr
    return msgpack.ExtType(code, data)


def role_for(uid: bytes, key: bytes) -> int:
    value = int.from_bytes(hashlib.sha256(ROLE_PREFIX + uid + b":" + key).digest()[:8], "big") / 2**64
    return 0 if value < .5 else 1 if value < .75 else 2 if value < .875 else 3


def read_user(connection, uid: str) -> dict:
    import msgpack
    blob = connection.execute("select data from federated_data where client_id=?", (uid.encode(),)).fetchone()[0]
    return msgpack.unpackb(zlib.decompress(blob), raw=False, ext_hook=ext)


def user_blocks(uid: str, data: dict):
    """Split one user's posts into the four roles: lists of (source index, text, label)."""
    blocks = [[], [], [], []]
    for j in range(len(data["tokens"])):
        typ = data["type"][j]
        if typ not in (b"question", b"answer"):
            raise ValueError("unknown type")
        label = 0 if typ == b"question" else 1
        key = b"|".join((data["creation_date"][j], data["title"][j], typ))
        blocks[role_for(uid.encode(), key)].append((j, data["tokens"][j].decode("utf8", "replace"), label))
    return blocks


def prepare(source: Path, out: Path, groups: list[int]) -> None:
    from sklearn.feature_extraction.text import HashingVectorizer

    m = manifest()
    vectorizer = HashingVectorizer(n_features=FEATURES, analyzer="word", ngram_range=(1, 2),
                                   alternate_sign=False, norm="l2")
    target = prepared_dir(out)
    target.mkdir(parents=True, exist_ok=True)
    wanted = {uid for g in groups for uid in m["groups"][g]}
    connection = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    for idx, uid in enumerate(sum(m["groups"], [])):
        if uid not in wanted:
            continue
        data = read_user(connection, uid)
        blocks = user_blocks(uid, data)
        meta = m["users"][uid]
        if len(data["tokens"]) != meta["source_count"] or [len(b) for b in blocks] != meta["role_counts"]:
            raise ValueError(f"user {uid}: record counts differ from the manifest")
        arrays = {}
        for role, block in enumerate(blocks):
            arrays[f"x{role}"] = vectorizer.transform([row[1] for row in block]).astype(np.float32).toarray()
            arrays[f"y{role}"] = np.array([row[2] for row in block], dtype=np.int64)
            arrays[f"id{role}"] = np.array([row[0] for row in block], dtype=np.int32)
        np.savez_compressed(target / f"user-{idx:02d}.npz", **arrays)
    connection.close()
    print("prepared", len(wanted), "users", flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="FedJAX stackoverflow_train.sqlite")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(1, 9)), choices=range(9))
    ap.add_argument("--skip-digest", action="store_true")
    args = ap.parse_args(argv)
    source = check_source(args.source, SOURCE_SHA, args.skip_digest)
    prepare(source, output_dir(args.out), args.groups)


if __name__ == "__main__":
    main()
