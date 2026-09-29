"""Record a script's results and compare them with the values reported in the paper.

Each script calls ``report(name, values)``. Values are written to
results/<name>.json and compared with expected/<name>.json, which lists every
quantity the paper reports together with an absolute tolerance that reflects
the precision printed in the paper.
"""
from __future__ import annotations

import json
import math

from .paths import EXPECTED, RESULTS


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def report(name: str, values: dict) -> bool:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(values, indent=1, sort_keys=True))
    exp_path = EXPECTED / f"{name}.json"
    if not exp_path.exists():
        print(f"[{name}] no expected values; results written")
        return True
    expected = json.loads(exp_path.read_text())
    got = _flatten(values)
    ok = True
    for key, spec in expected["checks"].items():
        want, tol = spec["value"], spec["tol"]
        have = got.get(key)
        good = have is not None and not (isinstance(have, float) and math.isnan(have)) and abs(have - want) <= tol
        ok &= good
        status = "ok  " if good else "FAIL"
        print(f"[{name}] {status} {key}: got {have!r}, paper {want} (tol {tol})  -- {spec.get('where', '')}")
    return ok
