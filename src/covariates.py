"""Train-only covariate correction for the 'standard' cross-individual target (hard rule 2).

Cross-individual expression is confounded by population structure (ancestry) and hidden/technical
factors. The 'standard' target residualizes the centered target on covariates:
    covariates = [genotype PCs (ancestry)] + [expression PCs (hidden/technical factors, a PEER proxy)]

CRITICAL: every covariate (the PCA fit AND the per-gene regression) is fit on **training individuals
only** and applied to held-out individuals. Fitting on the full cohort leaks population structure into
the test set (§3.5, §4.4). This module makes that non-negotiable by construction: `fit` sees only train.

Genotype PCs require genotype data (a gated fetch); until then the corrector runs with expression PCs
alone. The API is identical when genotype PCs are added — pass them in `extra_covariates`.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass
class ExpressionPCA:
    """Hidden-factor covariates from expression PCA. Fit on train individuals, project any."""
    k: int
    scaler: StandardScaler
    pca: PCA
    genes: pd.Index

    @classmethod
    def fit(cls, log_expr: pd.DataFrame, train_individuals: list[str], k: int = 20) -> "ExpressionPCA":
        """log_expr: genes × individuals. Fit scaler+PCA on train columns only (samples=individuals)."""
        X = log_expr[train_individuals].to_numpy().T          # (n_train, n_genes)
        scaler = StandardScaler().fit(X)
        pca = PCA(n_components=k, svd_solver="randomized", random_state=0).fit(scaler.transform(X))
        return cls(k=k, scaler=scaler, pca=pca, genes=log_expr.index)

    def transform(self, log_expr: pd.DataFrame, individuals: list[str]) -> pd.DataFrame:
        """Project individuals -> (individuals × k) PC-score covariates."""
        X = log_expr.loc[self.genes, individuals].to_numpy().T
        scores = self.pca.transform(self.scaler.transform(X))
        cols = [f"exprPC{i+1}" for i in range(self.k)]
        return pd.DataFrame(scores, index=individuals, columns=cols)


@dataclass
class CovariateCorrector:
    """Per-gene OLS residualization of a centered target on covariates. Fit on train only."""
    betas: np.ndarray          # (p_covariates+1, n_genes)  includes intercept row
    genes: pd.Index
    cov_cols: list

    @staticmethod
    def _design(cov: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(cov)), cov.to_numpy()])

    @classmethod
    def fit(cls, target: pd.DataFrame, covariates: pd.DataFrame,
            train_individuals: list[str]) -> "CovariateCorrector":
        """target: genes × individuals; covariates: individuals × p. Uses train individuals only."""
        C = cls._design(covariates.loc[train_individuals])           # (n_train, p+1)
        Y = target[train_individuals].to_numpy().T                   # (n_train, n_genes)
        betas, *_ = np.linalg.lstsq(C, Y, rcond=None)                # (p+1, n_genes)
        return cls(betas=betas, genes=target.index, cov_cols=list(covariates.columns))

    def transform(self, target: pd.DataFrame, covariates: pd.DataFrame,
                  individuals: list[str], restandardize: bool = True) -> pd.DataFrame:
        """Return residual target (genes × individuals) for `individuals` using train-fit betas."""
        C = self._design(covariates.loc[individuals])                # (n, p+1)
        Y = target.loc[self.genes, individuals].to_numpy().T         # (n, n_genes)
        resid = Y - C @ self.betas                                   # (n, n_genes)
        R = pd.DataFrame(resid.T, index=self.genes, columns=individuals)
        if restandardize:                                            # per-gene z across THESE individuals
            R = R.sub(R.mean(axis=1), axis=0).div(R.std(axis=1).replace(0, 1.0), axis=0)
        return R
