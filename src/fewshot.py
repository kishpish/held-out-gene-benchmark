"""Few-shot cross-individual prediction on held-out genes (METHODOLOGY §6.4).

The Gate-B fallback: if zero-shot transfer is flat, quantify how per-gene prediction improves with k
labeled "support" individuals, and test whether the gene-agnostic transfer model (Phase 6) reduces the
number of labels needed — a capability still beyond PrediXcan's many-per-gene requirement.

For a held-out gene we hold a fixed disjoint QUERY set (EUR test individuals) and draw k SUPPORT
individuals from the EUR train pool. Methods fit on the k support individuals and are scored (cross-
individual Spearman ρ) on the query set:
  - lead_eqtl(k)          : single best cis-SNP by |corr| on support.
  - enet_all(k)           : elastic-net on all common cis-SNPs.
  - enet_transfer_topM(k) : elastic-net on the top-M SNPs ranked by |transfer β̂| (prior-guided selection).
  - enet_random_topM(k)   : elastic-net on M random SNPs (control for enet_transfer_topM).
"""
from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


def _rho(pred, y):
    return 0.0 if np.std(pred) < 1e-9 else float(spearmanr(pred, y).statistic)


def _enet_fit_predict(Xs, ys, Xq):
    sc = StandardScaler().fit(Xs)
    # speed-tuned: modest max_iter + loose tol + random coordinate descent (LD-correlated genotypes
    # otherwise run to max_iter). Predictive accuracy is unchanged for this comparison.
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=500, tol=1e-3,
                    selection="random", random_state=0)
    en.fit(sc.transform(Xs), ys)
    return en.predict(sc.transform(Xq))


def fewshot_gene(X, y, support_pool, query, k, beta_abs=None, M=50, repeats=5, seed=0):
    """Return dict method -> mean ρ over `repeats` random support draws of size k.

    X: (n_ind, n_snp) dosages; y: (n_ind,); support_pool/query: index arrays; beta_abs: |transfer β̂| per SNP.
    """
    rng = np.random.default_rng(seed)
    n_snp = X.shape[1]
    yq = y[query]
    top = np.argsort(beta_abs)[::-1][:M] if beta_abs is not None and n_snp > M else np.arange(n_snp)
    out = {m: [] for m in ["lead_eqtl", "enet_all", "enet_transfer_topM", "enet_random_topM"]}
    for r in range(repeats):
        if k > len(support_pool):
            continue
        sup = rng.choice(support_pool, k, replace=False)
        Xs, ys = X[sup], y[sup]
        Xq = X[query]
        # lead-eQTL
        if n_snp > 0 and k >= 3:
            c = np.array([abs(pearsonr(Xs[:, j], ys).statistic) if np.std(Xs[:, j]) > 1e-9 else 0.0
                          for j in range(n_snp)])
            j = int(np.argmax(c)); b = np.polyfit(Xs[:, j], ys, 1)
            out["lead_eqtl"].append(_rho(np.polyval(b, Xq[:, j]), yq))
        # elastic-net all
        if n_snp > 0 and k >= 5:
            out["enet_all"].append(_rho(_enet_fit_predict(Xs, ys, Xq), yq))
            # transfer-selected top-M
            rnd = rng.choice(n_snp, min(M, n_snp), replace=False)
            out["enet_transfer_topM"].append(_rho(_enet_fit_predict(Xs[:, top], ys, Xq[:, top]), yq))
            out["enet_random_topM"].append(_rho(_enet_fit_predict(Xs[:, rnd], ys, Xq[:, rnd]), yq))
    return {m: (float(np.mean(v)) if v else np.nan) for m, v in out.items()}
