#!/usr/bin/env python3
"""Materialize the 30-group FEMNIST federations (eight writers each) from the Flower FEMNIST Parquet file.

Each writer's records are ordered by a fixed hash and split into four roles:
0 = local training (1/2), 1 = working inventory scored by providers (1/4),
2 = development target (1/8), 3 = external target (remainder).
The writer assignment is fixed in writer-manifest.json.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DEFAULT_OUT, DOWNLOADS, check_source, output_dir  # noqa: E402

MANIFEST = HERE / "writer-manifest.json"
SOURCE_SHA = "cdb389c8d41d5facab79a4a8d452724c30d9c629a1c1e0b1e9ae9de2c633522b"
DEFAULT_SOURCE = DOWNLOADS / "femnist-train.parquet"
# Fixed domain-separation string for the per-writer record order; changing it changes every split.
RECORD_SALT = "curation-record-v1:"
N_GROUPS = 34


def key(prefix: str, value: str) -> bytes:
    return hashlib.sha256((prefix + value).encode()).digest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def prepared_dir(out: Path) -> Path:
    return out / "femnist30" / "prepared"


def prepare(source: Path, out: Path, groups: list[int]) -> None:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from PIL import Image

    m = manifest()
    wanted = sorted({w for g in groups for w in m["groups"][g]})
    raw = pq.read_table(source, columns=["writer_id", "character", "image"])
    raw = raw.append_column("source_row", pa.array(np.arange(raw.num_rows, dtype=np.int64)))
    raw = raw.filter(pc.is_in(raw["writer_id"], value_set=pa.array(wanted)))
    cols = [raw[name].to_pylist() for name in ("writer_id", "character", "image", "source_row")]
    by_writer: dict[str, list] = defaultdict(list)
    for writer, label, item, row in zip(*cols):
        with Image.open(io.BytesIO(item["bytes"])) as im:
            pixels = np.asarray(im.convert("L"), dtype=np.uint8).copy()
        if pixels.shape != (28, 28) or not 0 <= label < 62:
            raise ValueError("bad image or label")
        by_writer[writer].append((int(row), pixels, int(label)))
    del cols, raw
    target = prepared_dir(out)
    target.mkdir(parents=True, exist_ok=True)
    for gi in groups:
        payload = {}
        for wi, writer in enumerate(m["groups"][gi]):
            records = sorted(by_writer[writer],
                             key=lambda item: (key(RECORD_SALT, f"{writer}:{item[0]}"), item[0]))
            n = len(records)
            if n != m["source_counts"][writer]:
                raise ValueError(f"writer {writer}: {n} records, manifest says {m['source_counts'][writer]}")
            a, b, c = n // 2, n // 2 + n // 4, n // 2 + n // 4 + n // 8
            roles = (records[:a], records[a:b], records[b:c], records[c:])
            if [len(r) for r in roles] != m["role_counts"][writer]:
                raise ValueError(f"writer {writer}: role sizes differ from the manifest")
            for ri, role in enumerate(roles):
                payload[f"w{wi}_r{ri}_x"] = np.stack([item[1] for item in role])
                payload[f"w{wi}_r{ri}_y"] = np.asarray([item[2] for item in role], dtype=np.int64)
                payload[f"w{wi}_r{ri}_rows"] = np.asarray([item[0] for item in role], dtype=np.int64)
        np.savez_compressed(target / f"group-{gi:02d}.npz", **payload)
        print("prepared group", gi, flush=True)


def load_group(out: Path, index: int):
    """Return (writers, players, parts); parts[writer][role] = (images, labels, source rows)."""
    m = manifest()
    group = m["groups"][index]
    with np.load(prepared_dir(out) / f"group-{index:02d}.npz") as payload:
        parts = {}
        for wi, writer in enumerate(group):
            parts[writer] = tuple((payload[f"w{wi}_r{ri}_x"].copy(),
                                   payload[f"w{wi}_r{ri}_y"].copy(),
                                   payload[f"w{wi}_r{ri}_rows"].copy()) for ri in range(4))
    return group, m["players"][index], parts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="Flower FEMNIST train Parquet file")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(N_GROUPS)))
    ap.add_argument("--skip-digest", action="store_true")
    args = ap.parse_args(argv)
    source = check_source(args.source, SOURCE_SHA, args.skip_digest)
    prepare(source, output_dir(args.out), args.groups)


if __name__ == "__main__":
    main()
