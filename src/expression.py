"""GEUVADIS expression preprocessing → centered cross-individual targets (METHODOLOGY §3.5–3.7).

Pipeline:
  1. load GD660 RPKM matrix (genes × libraries)
  2. collapse replicate libraries → one value per 1000G individual (mean)
  3. gene QC: keep genes expressed above a threshold in enough individuals
  4. rank-based inverse-normal transform (INT) per gene across individuals

The INT'd matrix IS the **centered cross-individual target** (per gene: mean ≈ 0, sd ≈ 1) — the
only axis we ever train/evaluate on (hard rule 4). This is the "light" target (ancestry retained).
The "standard" target additionally residualizes on covariates (genotype PCs + hidden factors);
that step is train-only (hard rule 2) and lives in `covariates.py`, applied after splits (Phase 4).

Notes:
- INT is rank-based, so an explicit log transform before it is a no-op (monotone) — we skip it for
  the target but keep a log-RPKM matrix for PCA/covariate modelling where magnitude matters.
- GEUVADIS `Coord` is hg19 and is discarded; GRCh38 TSS come from GENCODE (`annotation.py`).
"""
from __future__ import annotations
import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm


def load_gd660(gz: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (expr [genes × libraries] RPKM, gene_info [gene_id, gene_id_base, symbol])."""
    with gzip.open(gz, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    lib_cols = header[4:]
    usecols = [0, 1] + list(range(4, len(header)))
    df = pd.read_csv(gz, sep="\t", usecols=usecols)
    gene_info = pd.DataFrame({
        "gene_id": df["TargetID"],
        "gene_id_base": df["TargetID"].str.split(".").str[0],
        "symbol": df["Gene_Symbol"],
    })
    expr = df.set_index("TargetID")[lib_cols]
    expr.columns.name = "library"
    return expr, gene_info


def collapse_to_individuals(expr: pd.DataFrame) -> pd.DataFrame:
    """Average replicate libraries per 1000G individual (library id prefix before first '.')."""
    individuals = [c.split(".")[0] for c in expr.columns]
    grouped = expr.T.groupby(individuals).mean().T
    grouped.columns.name = "individual"
    return grouped


def gene_qc(expr_indiv: pd.DataFrame, min_rpkm: float = 0.1,
            min_frac: float = 0.5) -> pd.Index:
    """Genes expressed > min_rpkm in >= min_frac of individuals."""
    frac = (expr_indiv > min_rpkm).mean(axis=1)
    return expr_indiv.index[frac >= min_frac]


def inverse_normal_transform(matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-gene (per-row) rank-based INT across individuals (columns). Ties -> average rank.

    Maps each gene's values to standard-normal quantiles: z = Φ⁻¹((rank - 0.5) / n).
    Result per gene: mean ≈ 0, sd ≈ 1 — the centered cross-individual target.
    """
    vals = matrix.to_numpy(dtype=float)
    n = vals.shape[1]
    out = np.empty_like(vals)
    for i in range(vals.shape[0]):
        r = rankdata(vals[i], method="average")
        out[i] = norm.ppf((r - 0.5) / n)
    return pd.DataFrame(out, index=matrix.index, columns=matrix.columns)


def build_targets(gz: Path, min_rpkm: float = 0.1, min_frac: float = 0.5):
    """Full light-target pipeline. Returns dict with rpkm, log_rpkm, int (target), gene_info, qc genes."""
    expr, gene_info = load_gd660(gz)
    indiv = collapse_to_individuals(expr)
    keep = gene_qc(indiv, min_rpkm, min_frac)
    rpkm = indiv.loc[keep]
    log_rpkm = np.log2(rpkm + 1.0)
    target_int = inverse_normal_transform(rpkm)
    return {
        "rpkm": rpkm,                 # genes × individuals
        "log_rpkm": log_rpkm,
        "target_int": target_int,     # centered cross-individual target (light scheme)
        "gene_info": gene_info.set_index("gene_id").loc[keep],
        "genes_kept": keep,
        "n_individuals": rpkm.shape[1],
    }
