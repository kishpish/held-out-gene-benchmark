"""Linear cross-individual baselines (METHODOLOGY §5.1) — the models to beat / reproduce the gap.

All operate per gene on cis-window genotypes and are trained on TRAIN individuals, evaluated on
HELD-OUT individuals of the SAME gene (the {gene present, held-out individual} cell). The scientific
point (§5): these per-gene linear models are **undefined on a truly held-out gene** (no labels to fit) —
the capability gap our method targets. Here we quantify how strong they are where they *are* defined.

Baselines:
  - mean            — predict the population mean (0 for the centered target) → cross-individual ρ ≈ 0 (floor).
  - lead-eQTL       — single best cis-SNP chosen on TRAIN (max |Pearson r| with y), linear fit.
  - elastic-net     — PrediXcan/FUSION-style ElasticNetCV on all cis SNPs (train) → predict test.

Evaluation is within-ancestry (EUR) to keep the comparison about cis-grammar, not ancestry, and comparable
to the EUR cis-h² ceiling from Phase 3. No test-individual information is used in fitting/selection.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

from personal import stream_dosage_matrix


def cis_dosage(vcf_url, chrom, tss1, samples, sample_file, length=196_608, maf_min=0.01):
    """(X [n_samp, n_snp] mean-imputed standardized-ready dosages, af, positions) for a gene window."""
    half = length // 2
    pos, D, af = stream_dosage_matrix(vcf_url, chrom, tss1 - half, tss1 + half - 1, samples, sample_file)
    if D.shape[0] == 0:
        return np.zeros((len(samples), 0)), af, pos
    keep = (af >= maf_min) & (af <= 1 - maf_min)
    D = D[keep]; af = af[keep]; pos = pos[keep]
    # mean-impute missing per SNP
    cm = np.nanmean(D, axis=1, keepdims=True)
    nanx = np.where(np.isnan(D)); D[nanx] = np.take(cm, nanx[0])
    return D.T.astype(np.float64), af, pos       # X: individuals × SNPs


@dataclass
class BaselineResult:
    gene_id: str
    n_snp: int
    rho_lead: float
    rho_enet: float
    rho_mean: float
    r_lead: float
    r_enet: float
    enet_nonzero: int


def _safe_spearman(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(spearmanr(a, b).statistic)


def evaluate_gene(gene_id, X, y, train_idx, test_idx) -> BaselineResult:
    """Fit lead-eQTL and elastic-net on train rows, score cross-individual ρ on test rows."""
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    n_snp = X.shape[1]

    # lead-eQTL: best SNP by |Pearson r| on train
    rho_lead = r_lead = 0.0
    if n_snp > 0:
        corrs = np.array([abs(pearsonr(Xtr[:, j], ytr).statistic) if np.std(Xtr[:, j]) > 1e-9 else 0.0
                          for j in range(n_snp)])
        j = int(np.argmax(corrs))
        b = np.polyfit(Xtr[:, j], ytr, 1)
        pred = np.polyval(b, Xte[:, j])
        rho_lead = _safe_spearman(pred, yte)
        r_lead = float(pearsonr(pred, yte).statistic) if np.std(pred) > 1e-9 else 0.0

    # elastic-net (PrediXcan-style) on all cis SNPs; standardize SNPs (train-fit).
    # sklearn>=1.9: ElasticNetCV takes `alphas` (int = #alphas), not `n_alphas`.
    rho_enet = r_enet = 0.0; nz = 0
    if n_snp > 0:
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        # PrediXcan-style l1_ratio=0.5; speed-tuned CV (random coordinate descent).
        en = ElasticNetCV(l1_ratio=0.5, alphas=20, cv=3, max_iter=2000,
                          selection="random", random_state=0, n_jobs=1)
        en.fit(Xtr_s, ytr)
        pred = en.predict(Xte_s)
        rho_enet = _safe_spearman(pred, yte)
        r_enet = float(pearsonr(pred, yte).statistic) if np.std(pred) > 1e-9 else 0.0
        nz = int((en.coef_ != 0).sum())

    return BaselineResult(gene_id=gene_id, n_snp=n_snp, rho_lead=rho_lead, rho_enet=rho_enet,
                          rho_mean=0.0, r_lead=r_lead, r_enet=r_enet, enet_nonzero=nz)
