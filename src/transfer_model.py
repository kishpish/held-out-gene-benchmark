"""Gene-agnostic transferable variant-effect model (METHODOLOGY §6.1, P1+P3).

The novel method's CPU-tractable v1. A model learns, on TRAINING-chromosome genes, a map from
gene-agnostic local features (signed TF-motif-disruption deltas + distance-to-TSS + MAF) to a variant's
effect on expression. Because the model never sees gene identity — only local sequence context — it can
be applied ZERO-SHOT to held-out-chromosome genes: for each cis-variant predict an effect β̂ from its
features, then form each individual's predicted expression deviation as Σ dosage · β̂, and correlate across
individuals with the centered target.

This decouples the transferable variant-effect function from the (non-transferable) gene baseline (P1)
and carries direction via signed motif deltas (P3). It is exactly the capability per-gene linear models
lack: prediction on a gene with zero training labels.

Label for training: the per-variant marginal cis-eQTL effect (OLS slope of centered expression on dosage,
train individuals) — a noisy but abundant supervised signal averaged across many genes.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

import motifs as motif_mod


def variant_features(ref, chrom, pos1, ref_allele, alt_allele, tss1, strand,
                     motif_list, maf, flank=19) -> np.ndarray | None:
    """Feature vector for one variant: [signed motif Δ ×K, |Δ| ×K, signed log-dist-to-TSS, 1/(1+|d|/1e3), maf]."""
    try:
        ref_loc, alt_loc = motif_mod.local_windows(ref, chrom, pos1, ref_allele, alt_allele, flank)
    except Exception:
        return None
    deltas = motif_mod.variant_deltas(ref_loc, alt_loc, motif_list)     # (K,)
    d = (pos1 - tss1) * (1 if strand == "+" else -1)                    # signed TSS-relative distance
    dist_feats = np.array([np.sign(d) * np.log1p(abs(d)), 1.0 / (1.0 + abs(d) / 1000.0)], dtype=np.float32)
    return np.concatenate([deltas, np.abs(deltas), dist_feats, [np.float32(maf)]])


def variant_features_win(ref_window: str, window_start0: int, pos1, ref_allele, alt_allele,
                         tss1, strand, motif_list, maf, flank=19) -> np.ndarray | None:
    """Same features as variant_features but slices the local window from an in-memory reference
    window string (fast: no per-variant FASTA seek). window_start0 is the 0-based start of ref_window."""
    off = (pos1 - 1) - window_start0
    if off - flank < 0 or off + flank + 1 > len(ref_window):
        return None
    ref_loc = ref_window[off - flank: off + flank + 1]
    if ref_loc[flank] != ref_allele:
        return None
    alt_loc = ref_loc[:flank] + alt_allele + ref_loc[flank + 1:]
    deltas = motif_mod.variant_deltas(ref_loc, alt_loc, motif_list)
    d = (pos1 - tss1) * (1 if strand == "+" else -1)
    dist_feats = np.array([np.sign(d) * np.log1p(abs(d)), 1.0 / (1.0 + abs(d) / 1000.0)], dtype=np.float32)
    return np.concatenate([deltas, np.abs(deltas), dist_feats, [np.float32(maf)]])


@dataclass
class TransferModel:
    model: object
    n_features: int

    @classmethod
    def train(cls, X: np.ndarray, y: np.ndarray, kind: str = "gbm") -> "TransferModel":
        if kind == "gbm":
            from sklearn.ensemble import HistGradientBoostingRegressor
            m = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                              max_iter=400, l2_regularization=1.0,
                                              early_stopping=True, random_state=0)
        else:
            from sklearn.linear_model import RidgeCV
            m = RidgeCV(alphas=np.logspace(-2, 3, 20))
        m.fit(X, y)
        return cls(model=m, n_features=X.shape[1])

    def predict_effects(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def aggregate_individual_scores(dosage: np.ndarray, beta_hat: np.ndarray) -> np.ndarray:
    """Per-individual predicted deviation = Σ_variant dosage · β̂.  dosage: (n_var, n_samp)."""
    d = np.nan_to_num(dosage, nan=0.0)
    return (beta_hat[:, None] * d).sum(axis=0)      # (n_samp,)
