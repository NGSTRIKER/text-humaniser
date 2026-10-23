"""Score → probability conversion and reproducible seeding (Stage 6, plan §8).

Randomness only ever chooses among candidates that already passed every
hard gate; these helpers never see rejected candidates.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Literal, Sequence

Method = Literal["softmax", "linear", "uniform"]


def selection_probabilities(
    scores: Sequence[float], method: Method = "softmax", temperature: float = 0.5
) -> list[float]:
    """Return probabilities (summing to 1) for ``scores``.

    * ``softmax``: ``exp((s - max) / T)`` — numerically stable; low T ≈ greedy.
    * ``linear``: proportional to ``s - min + ε`` (normalized nonnegative weights).
    * ``uniform``: every acceptable candidate equally likely.
    """
    n = len(scores)
    if n == 0:
        return []
    if method == "uniform":
        return [1.0 / n] * n
    if method == "linear":
        lo = min(scores)
        weights = [s - lo + 1e-3 for s in scores]
    elif method == "softmax":
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        top = max(scores)
        weights = [math.exp((s - top) / temperature) for s in scores]
    else:
        raise ValueError(f"unknown selection method {method!r}")
    total = sum(weights)
    return [w / total for w in weights]


def derive_seed(seed: int | None, *keys: object) -> int | None:
    """Stable sub-seed for a part of the job (e.g. a chunk index).

    Uses SHA-256 rather than ``hash()`` so it does not change between
    Python processes.  ``None`` stays ``None`` (non-reproducible run).
    """
    if seed is None:
        return None
    material = ":".join([str(seed), *map(str, keys)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def make_rng(seed: int | None, *keys: object) -> random.Random:
    return random.Random(derive_seed(seed, *keys))


def weighted_choice(items: Sequence, probabilities: Sequence[float], rng: random.Random):
    """One item drawn according to ``probabilities`` using ``rng``."""
    return rng.choices(list(items), weights=list(probabilities), k=1)[0]
