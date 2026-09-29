"""Check every file under data/ against data/SHA256SUMS.

    python -m auditworth.verify_data          (run with src/ on PYTHONPATH)
    python -m auditworth.verify_data --write  (regenerate the list; maintainers only)
"""
from __future__ import annotations

import hashlib
import sys

from .paths import DATA

SUMS = DATA / "SHA256SUMS"


def files():
    return sorted(p for p in DATA.rglob("*") if p.is_file() and p.name not in ("SHA256SUMS", "README.md"))


def digest(p) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--write" in argv:
        SUMS.write_text("".join(f"{digest(p)}  {p.relative_to(DATA).as_posix()}\n" for p in files()))
        print(f"wrote {SUMS}")
        return 0
    listed = {}
    for line in SUMS.read_text().splitlines():
        h, name = line.split("  ", 1)
        listed[name] = h
    bad = [n for n, h in listed.items() if not (DATA / n).is_file() or digest(DATA / n) != h]
    extra = [p.relative_to(DATA).as_posix() for p in files() if p.relative_to(DATA).as_posix() not in listed]
    print(f"{len(listed)} files listed, {len(bad)} mismatched or missing, {len(extra)} unlisted")
    for n in bad[:20]:
        print("  mismatch:", n)
    for n in extra[:20]:
        print("  unlisted:", n)
    return 0 if not bad and not extra else 1


if __name__ == "__main__":
    sys.exit(main())
