#!/usr/bin/env python
"""Phase 1 — leakage/correctness test for the train-only covariate corrector.

Documented in docs/PHASE1_expression_targets.md. Proves that CovariateCorrector, fit on TRAIN
individuals only, removes a confounder on HELD-OUT individuals (generalization, no leakage) while
preserving confounder-independent signal. This is the machinery that will produce the 'standard'
target after Phase-4 splits; verifying it now guards hard rule 2.

Run: .venv/bin/python scripts/phase1_test_covariates.py   (exits 0 on pass)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd

from covariates import CovariateCorrector, ExpressionPCA


def main() -> None:
    rng = np.random.default_rng(0)
    n_ind, n_genes = 300, 200
    inds = [f"S{i:03d}" for i in range(n_ind)]
    train, test = inds[:200], inds[200:]

    # Confounder c (ancestry-like), per individual. Gene-specific loading beta_g.
    c = pd.Series(rng.normal(size=n_ind), index=inds, name="conf")
    beta = rng.normal(scale=2.0, size=n_genes)
    true_signal = rng.normal(size=(n_genes, n_ind))          # confounder-independent
    noise = rng.normal(scale=0.3, size=(n_genes, n_ind))
    target = pd.DataFrame(beta[:, None] * c.values[None, :] + true_signal + noise,
                          index=[f"g{j}" for j in range(n_genes)], columns=inds)
    cov = c.to_frame("conf")

    # Fit on TRAIN only, apply to TEST.
    corr = CovariateCorrector.fit(target, cov, train)
    resid_test = corr.transform(target, cov, test, restandardize=False)

    # (A) confounder correlation on TEST collapses after correction
    def mean_abs_corr(mat, series, cols):
        cc = series.loc[cols].values
        return float(np.mean([abs(np.corrcoef(mat.loc[g, cols].values, cc)[0, 1])
                              for g in mat.index]))

    before = mean_abs_corr(target, c, test)
    after = mean_abs_corr(resid_test, c, test)
    # empirical null floor: |corr| of residual vs a shuffled confounder at this sample size
    c_shuf = pd.Series(rng.permutation(c.loc[test].values), index=test)
    null_floor = mean_abs_corr(resid_test, c_shuf, test)
    print(f"[A] mean |corr(gene, confounder)| on TEST: before={before:.3f}  after={after:.3f}  "
          f"(null floor≈{null_floor:.3f})")
    assert before > 0.4, "confounder should be strong before correction"
    assert after < before / 4 and after < null_floor * 1.6, \
        "confounder not removed to ~null on held-out individuals"

    # (B) confounder-independent signal preserved (residual still tracks true_signal on TEST)
    ts = pd.DataFrame(true_signal, index=target.index, columns=inds)
    keep_corr = float(np.mean([abs(np.corrcoef(resid_test.loc[g, test].values,
                                               ts.loc[g, test].values)[0, 1]) for g in target.index]))
    print(f"[B] mean |corr(residual, true_signal)| on TEST: {keep_corr:.3f}  (should stay high)")
    assert keep_corr > 0.7, "real signal was destroyed"

    # (C) ExpressionPCA fit on train projects test deterministically (no refit / no peeking)
    log_expr = target  # reuse as a stand-in expression matrix
    ep = ExpressionPCA.fit(log_expr, train, k=5)
    s1 = ep.transform(log_expr, test)
    s2 = ep.transform(log_expr, test)
    assert np.allclose(s1.values, s2.values), "PCA projection not deterministic"
    # train scores from stored PCA match a direct fit_transform on train
    direct = ep.pca.transform(ep.scaler.transform(log_expr[train].to_numpy().T))
    assert np.allclose(direct, ep.transform(log_expr, train).values), "train projection inconsistent"
    print("[C] ExpressionPCA train-fit projection is deterministic and consistent")

    print("\nALL COVARIATE TESTS PASSED")


if __name__ == "__main__":
    main()
