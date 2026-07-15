"""TF motif scoring for transferable variant-effect features (METHODOLOGY §6.2, P3).

Loads JASPAR PFMs → log-odds PWMs and computes, for each variant, a **signed motif-disruption delta**
per motif: how much the alternate allele strengthens (+) or weakens (−) the best local match to that TF's
motif. These deltas are the gene-agnostic "grammar" features the Phase-6 model learns to map to expression
effects — with sign, which is exactly what direction-of-effect transfer requires (Sasse 2023).

A variant's motif deltas depend only on the local reference sequence and the ref/alt alleles — NOT on the
individual — so they are computed once per variant and reused across all individuals.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genome import revcomp

_BASE = {"A": 0, "C": 1, "G": 2, "T": 3}


@dataclass
class PWM:
    name: str
    matrix: np.ndarray      # (L, 4) log-odds
    length: int

    @classmethod
    def from_jaspar(cls, path: Path, pseudocount: float = 0.8) -> "PWM":
        d = json.loads(Path(path).read_text())
        pfm = d["pfm"]
        counts = np.array([pfm["A"], pfm["C"], pfm["G"], pfm["T"]], dtype=float).T  # (L,4)
        col = counts.sum(axis=1, keepdims=True)
        freq = (counts + pseudocount) / (col + 4 * pseudocount)
        logodds = np.log2(freq / 0.25)
        return cls(name=d["name"], matrix=logodds, length=counts.shape[0])


def load_motifs(jaspar_dir: Path) -> list[PWM]:
    return [PWM.from_jaspar(p) for p in sorted(Path(jaspar_dir).glob("*.json"))]


def _seq_to_idx(seq: str) -> np.ndarray:
    return np.array([_BASE.get(b, -1) for b in seq.upper()], dtype=int)


def best_score(seq: str, pwm: PWM) -> float:
    """Max PWM log-odds over all windows of `seq`, both strands (vectorized)."""
    L = pwm.length
    best = -1e9
    ar = np.arange(L)
    for s in (seq, revcomp(seq)):
        idx = _seq_to_idx(s)
        n = len(idx)
        if n < L:
            continue
        win = np.lib.stride_tricks.sliding_window_view(idx, L)      # (n-L+1, L)
        valid = (win >= 0).all(axis=1)
        if not valid.any():
            continue
        w = win[valid]
        scores = pwm.matrix[ar[None, :], w].sum(axis=1)             # (n_valid,)
        best = max(best, float(scores.max()))
    return best if best > -1e8 else 0.0


def variant_deltas(ref_local: str, alt_local: str, motifs: list[PWM]) -> np.ndarray:
    """Signed Δ (alt_best − ref_best) per motif for a local window centered on the variant."""
    out = np.empty(len(motifs), dtype=np.float32)
    for k, m in enumerate(motifs):
        out[k] = best_score(alt_local, m) - best_score(ref_local, m)
    return out


def local_windows(ref, chrom: str, pos1: int, ref_allele: str, alt_allele: str,
                  flank: int) -> tuple[str, str]:
    """Reference and alt local sequence windows [pos-flank, pos+flank] (1bp SNV substitution)."""
    start0 = pos1 - 1 - flank
    end0 = pos1 - 1 + flank + 1
    ref_seq = ref.get_seq(chrom, start0, end0)
    j = flank  # variant index within window
    assert ref_seq[j] == ref_allele, f"ref mismatch @ {chrom}:{pos1}: {ref_seq[j]} vs {ref_allele}"
    alt_seq = ref_seq[:j] + alt_allele + ref_seq[j + 1:]
    return ref_seq, alt_seq
