#!/usr/bin/env python3
"""Materialize the four 64-writer FEMNIST groups of the update-scaling study from the Flower FEMNIST Parquet file.

Each writer's records are ordered by a fixed hash and split into four roles:
0 = local training (1/2), 1 = working inventory (1/4), 2 = development target (1/8),
3 = external target (remainder). The writer assignment is fixed in client-manifest.json
(group 0 was used for model design only; groups 1-4 are analysed).
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

MANIFEST = HERE / "client-manifest.json"
SOURCE_SHA = "cdb389c8d41d5facab79a4a8d452724c30d9c629a1c1e0b1e9ae9de2c633522b"
DEFAULT_SOURCE = DOWNLOADS / "femnist-train.parquet"
# Fixed domain-separation string for the per-writer record order; changing it changes every split.
RECORD_SALT = "curation-record-v1:"


def digest_key(prefix, value):
    return hashlib.sha256((prefix + value).encode()).digest()


def manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def prepared_dir(out: Path) -> Path:
    return out / "femnist_scaled" / "prepared"


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
    cols = [raw[c].to_pylist() for c in ("writer_id", "character", "image", "source_row")]
    by = defaultdict(list)
    for w, label, item, row in zip(*cols):
        with Image.open(io.BytesIO(item["bytes"])) as im:
            gray = np.asarray(im.convert("L"), dtype=np.uint8).copy()
        if gray.shape != (28, 28) or not 0 <= label < 62:
            raise ValueError("bad image or label")
        by[w].append((int(row), gray, int(label)))
    del cols, raw
    target = prepared_dir(out)
    target.mkdir(parents=True, exist_ok=True)
    for g in groups:
        payload = {}
        for wi, w in enumerate(m["groups"][g]):
            rec = sorted(by[w], key=lambda r: (digest_key(RECORD_SALT, f"{w}:{r[0]}"), r[0]))
            n = len(rec)
            a = n // 2
            b = a + n // 4
            c = b + n // 8
            blocks = (rec[:a], rec[a:b], rec[b:c], rec[c:])
            expected = m["users"][w]
            if n != expected["n"] or list(map(len, blocks)) != expected["role_counts"]:
                raise ValueError(f"writer {w}: record counts differ from the manifest")
            for ri, block in enumerate(blocks):
                rows = np.asarray([r[0] for r in block], np.int64)
                if hashlib.sha256(rows.tobytes()).hexdigest() != expected["role_source_row_sha256"][ri]:
                    raise ValueError(f"writer {w}: role {ri} rows differ from the manifest")
                payload[f"w{wi}_r{ri}_x"] = np.stack([r[1] for r in block])
                payload[f"w{wi}_r{ri}_y"] = np.asarray([r[2] for r in block], np.int64)
                payload[f"w{wi}_r{ri}_rows"] = rows
        np.savez_compressed(target / f"group-{g}.npz", **payload)
        print("prepared group", g, flush=True)


def load_group(out: Path, group: int):
    """Return (writers, players, parts); parts[writer][role] = (images, labels, source rows)."""
    m = manifest()
    writers = m["groups"][group]
    with np.load(prepared_dir(out) / f"group-{group}.npz") as z:
        parts = {w: tuple((z[f"w{wi}_r{ri}_x"].copy(), z[f"w{wi}_r{ri}_y"].copy(), z[f"w{wi}_r{ri}_rows"].copy())
                          for ri in range(4)) for wi, w in enumerate(writers)}
    return writers, m["players"][group], parts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="Flower FEMNIST train Parquet file")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output root (default: training/output)")
    ap.add_argument("--groups", type=int, nargs="+", default=[1, 2, 3, 4], choices=range(5))
    ap.add_argument("--skip-digest", action="store_true")
    args = ap.parse_args(argv)
    source = check_source(args.source, SOURCE_SHA, args.skip_digest)
    prepare(source, output_dir(args.out), args.groups)


if __name__ == "__main__":
    main()
