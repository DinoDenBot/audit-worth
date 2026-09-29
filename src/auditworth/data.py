"""Loaders for the cached per-record scores in data/.

Every cache stores, for each evaluation provider (client) p, the 0/1 correctness
of every candidate action on each record of p's complete working inventory
(``bits``) and per-record Shapley values (``psi``). See data/README.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import DATA

# FEMNIST 30-group study: groups 4-33 are analysed, 0-3 were reserved for debugging.
FEMNIST30_GROUPS = tuple(range(4, 34))
FEMNIST30_SEEDS = (2, 3, 4, 5)
DECAY_ROUNDS = (80, 81, 82, 84, 88, 96, 104, 112, 120)


def _femnist_state(path: Path) -> list[dict]:
    with np.load(path) as d:
        out = []
        for p in range(4):
            rows = d[f"rows_{p}"]
            order = np.argsort(rows, kind="stable")
            bits = d[f"bits_{p}"][:, order].astype(float)
            psi = np.stack([d[f"psi_player{j}_provider{p}"][order] for j in range(4)])
            out.append({"rows": rows[order], "bits": bits, "psi": psi})
    return out


def femnist30_state(group: int, rnd: int, seed: int, epochs: int = 1, role: str = "confirmation") -> list[dict]:
    """Round-``rnd`` state of a 30-group FEMNIST federation (rounds 80, 120, 160).

    ``bits`` has shape (5, n): keep-current (row 0) and the four contributors'
    updates; ``psi`` has shape (4, n): each contributor's per-record Shapley value.
    """
    suffix = "" if epochs == 1 else f"-e{epochs}"
    return _femnist_state(DATA / "femnist30" / "state" / f"g{group:02d}-r{rnd}-s{seed}{suffix}-{role}.npz")


def femnist30_decay_state(group: int, rnd: int, seed: int) -> list[dict]:
    """Deterministically replayed later-round state (rounds in DECAY_ROUNDS)."""
    return _femnist_state(DATA / "femnist30" / "decay" / f"g{group:02d}-r{rnd}-s{seed}-confirmation.npz")


def femnist_scaled_state(group: int, scale: str, seed: int, state: int) -> dict:
    """Update-scaling FEMNIST contract: four own-update candidates scaled by 0.5/1/2."""
    with np.load(DATA / "femnist_scaled" / "state" / f"g{group}-x{scale}-z{seed}-a{state}.npz") as d:
        return {k: d[k] for k in d.files}


def har_state(seed: int, state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """UCI HAR: per provider (bits (4, n), own Shapley psi (n,))."""
    with np.load(DATA / "har" / "state" / f"s{seed:02d}-a{state}.npz") as d:
        return [(d[f"bits_{p}"].astype(float), d[f"psi_{p}"].astype(float)) for p in range(4)]


def stackoverflow_state(group: int, seed: int, state: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stack Overflow (trained MLP): per provider (bits (4, n), own Shapley psi (n,))."""
    with np.load(DATA / "stackoverflow" / f"group-{group}" / f"score-s{seed:02d}-a{state}.npz") as z:
        return [(z[f"bits_{p}"].astype(float), z[f"psi_{p}"][p].astype(float)) for p in range(4)]


def stackoverflow_strong_state(group: int, seed: int, state: int) -> dict:
    """Stack Overflow with the stronger 120-round sparse-feature model."""
    with np.load(DATA / "stackoverflow_strong" / f"group-{group}" / f"score-s{seed:02d}-a{state}.npz") as z:
        return {k: z[k] for k in z.files}
