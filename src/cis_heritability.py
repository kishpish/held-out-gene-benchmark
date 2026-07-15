"""cis-heritability estimation (METHODOLOGY §3.8) — the per-gene correlation ceiling.

For each gene we estimate h²_cis: the fraction of (centered) expression variance explained by local
genotype. √h²_cis is the theoretical ceiling on per-gene cross-individual correlation, used to (i) set
honest go/no-go thresholds, (ii) report 'fraction of cis-h² explained', and (iii) SELECT evaluation
genes (drop genes with no learnable signal).

We report the ceiling **within the Enformer receptive field** (196,608 bp window, TSS-centered) — i.e.
the heritability attributable to variants the model can actually see — which is the honest ceiling for
our model. (A wider ±1 Mb cis estimate can be added for context.)

Model (single-component GREML, as in GCTA):
    y = Xβ + g + e,   g ~ N(0, σ²_g K),   e ~ N(0, σ²_e I),   K = standardized-genotype GRM.
h² = σ²_g / (σ²_g + σ²_e). Estimated by REML via eigendecomposition of K (n≈449 → fast). A likelihood-
ratio test vs σ²_g = 0 gives a p-value for 'has cis-h²' (0.5·χ²₁ boundary mixture). Haseman-Elston
regression provides a fast moment-estimator cross-check.

Ancestry note: the cis-GRM can be inflated by population structure. Primary estimates are computed within
the homogeneous EUR subset; full-cohort estimates are secondary (documented caveat). Genotype-PC fixed
effects can be added to X when available.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2


def build_grm(dosage: np.ndarray, maf_min: float = 0.01):
    """dosage: (n_snp, n_samp) with NaN missing. Returns (GRM (n_samp,n_samp), n_snp_used).

    Mean-impute missing, filter MAF>=maf_min, standardize each SNP to mean 0 / var 1 under HWE,
    GRM = Z Zᵀ / M with mean diagonal normalized to 1.
    """
    D = dosage.copy()
    af = np.nanmean(D, axis=1) / 2.0
    keep = (af >= maf_min) & (af <= 1 - maf_min)
    D = D[keep]; af = af[keep]
    if D.shape[0] == 0:
        n = dosage.shape[1]
        return np.eye(n), 0
    # mean-impute missing
    col_mean = np.nanmean(D, axis=1, keepdims=True)
    inds = np.where(np.isnan(D))
    D[inds] = np.take(col_mean, inds[0])
    # standardize under HWE: (g - 2p)/sqrt(2p(1-p))
    sd = np.sqrt(2 * af * (1 - af))[:, None]
    Z = (D - 2 * af[:, None]) / sd
    M = Z.shape[0]
    K = (Z.T @ Z) / M
    K /= np.mean(np.diag(K))          # normalize mean diagonal to 1
    return K, int(M)


@dataclass
class H2Result:
    h2: float
    se: float
    pval: float
    sigma_g: float
    sigma_e: float
    n_snp: int
    n_ind: int
    method: str = "REML"


def _reml_negll(delta: float, lam: np.ndarray, yt: np.ndarray, Xt: np.ndarray):
    """-REML log-likelihood at variance ratio delta=σ²_g/σ²_e (profiled over σ²_e and β)."""
    n, p = Xt.shape
    d = delta * lam + 1.0
    Xtd = Xt / d[:, None]
    XtX = Xt.T @ Xtd
    Xty = Xtd.T @ yt
    beta = np.linalg.solve(XtX, Xty)
    r = yt - Xt @ beta
    rss = np.sum(r * r / d)
    sign, logdetXtX = np.linalg.slogdet(XtX)
    sigma_e = rss / (n - p)
    ll = -0.5 * (np.sum(np.log(d)) + (n - p) * np.log(sigma_e) + logdetXtX)
    return -ll


def reml_h2(y: np.ndarray, K: np.ndarray, X: np.ndarray | None = None) -> H2Result:
    """REML h² for one gene. y: (n,), K: (n,n) GRM, X: (n,p) fixed effects (default intercept)."""
    n = len(y)
    if X is None:
        X = np.ones((n, 1))
    lam, U = np.linalg.eigh(K)
    lam = np.clip(lam, 0, None)
    yt = U.T @ y
    Xt = U.T @ X
    res = minimize_scalar(_reml_negll, bounds=(1e-6, 1e6), method="bounded",
                          args=(lam, yt, Xt), options={"xatol": 1e-5})
    delta = float(res.x)
    ll_alt = -res.fun
    # σ²_e at optimum
    p = Xt.shape[1]
    d = delta * lam + 1.0
    Xtd = Xt / d[:, None]
    beta = np.linalg.solve(Xt.T @ Xtd, Xtd.T @ yt)
    r = yt - Xt @ beta
    sigma_e = float(np.sum(r * r / d) / (n - p))
    sigma_g = delta * sigma_e
    h2 = sigma_g / (sigma_g + sigma_e)
    # LRT vs δ=0
    ll_null = -_reml_negll(1e-6, lam, yt, Xt)
    lrt = max(0.0, 2 * (ll_alt - ll_null))
    pval = 0.5 * chi2.sf(lrt, df=1)          # boundary mixture 0.5·χ²₁
    # SE via numerical curvature of profile LL in h2-space (delta method, rough)
    se = _h2_se(delta, lam, yt, Xt, sigma_e)
    return H2Result(h2=float(h2), se=float(se), pval=float(pval),
                    sigma_g=float(sigma_g), sigma_e=float(sigma_e),
                    n_snp=0, n_ind=n)


def _h2_se(delta, lam, yt, Xt, sigma_e, eps=1e-3):
    """Rough SE of h² from the curvature of the REML profile likelihood in δ (delta method)."""
    f0 = _reml_negll(delta, lam, yt, Xt)
    fp = _reml_negll(delta * (1 + eps) + eps, lam, yt, Xt)
    fm = _reml_negll(max(1e-6, delta * (1 - eps) - eps), lam, yt, Xt)
    h = delta * eps + eps
    curv = (fp - 2 * f0 + fm) / (h * h)       # d²(-LL)/dδ²
    var_delta = 1.0 / curv if curv > 0 else np.nan
    # h2 = δ/(δ+1); dh2/dδ = 1/(δ+1)²
    g = 1.0 / (delta + 1) ** 2
    return float(np.sqrt(var_delta) * g) if np.isfinite(var_delta) else float("nan")


def he_regression_h2(y: np.ndarray, K: np.ndarray) -> float:
    """Haseman-Elston moment estimator (fast cross-check). y standardized."""
    n = len(y)
    ys = (y - y.mean()) / y.std()
    iu = np.triu_indices(n, k=1)
    prod = np.outer(ys, ys)[iu]
    k = K[iu]
    b = np.polyfit(k, prod, 1)[0]             # slope ≈ h²
    return float(np.clip(b, 0, 1))
