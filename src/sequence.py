"""Sequence one-hot encoding (kipoiseq/Enformer convention: A,C,G,T; N -> all-zero)."""
from __future__ import annotations
import numpy as np

_BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot(seq: str) -> np.ndarray:
    """(len, 4) float32 one-hot; unknown bases -> zero row. Uppercases input."""
    s = seq.upper()
    arr = np.zeros((len(s), 4), dtype=np.float32)
    for i, b in enumerate(s):
        j = _BASE_IDX.get(b)
        if j is not None:
            arr[i, j] = 1.0
    return arr
