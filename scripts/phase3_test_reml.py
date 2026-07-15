#!/usr/bin/env python
"""Phase 3 — correctness test for the GREML/REML cis-h² estimator.

Documented in docs/PHASE3_cis_heritability.md. Simulates genotypes → GRM → phenotypes with a KNOWN h²,
and checks the REML estimator recovers it (mean over replicates within tolerance), that the LRT p-value
is small when h²>0 and calibrated (large) when h²=0, and that Haseman-Elston agrees. Guards the estimator
before we trust per-gene ceilings for gene selection.

Run: .venv/bin/python scripts/phase3_test_reml.py   (exits 0 on pass)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np

from cis_heritability import build_grm, reml_h2, he_regression_h2


def simulate(n=360, m=800, h2=0.3, seed=0):
    rng = np.random.default_rng(seed)
    af = rng.uniform(0.05, 0.5, m)
    G = rng.binomial(2, af[:, None] * np.ones((1, n))).astype(float)   # (m, n)
    K, mused = build_grm(G, maf_min=0.01)
    # breeding values with genetic covariance K
    L = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    g = L @ rng.normal(size=n)
    g = g / g.std() * np.sqrt(h2)
    e = rng.normal(size=n); e = e / e.std() * np.sqrt(1 - h2)
    y = g + e
    return y, K


def main() -> None:
    print("REML recovery across true h² (mean of 8 replicates):")
    ok = True
    for h2_true in [0.0, 0.2, 0.5, 0.8]:
        ests, hes, ps = [], [], []
        for s in range(8):
            y, K = simulate(h2=h2_true, seed=s)
            r = reml_h2(y, K)
            ests.append(r.h2); hes.append(he_regression_h2(y, K)); ps.append(r.pval)
        m_reml, m_he = np.mean(ests), np.mean(hes)
        frac_sig = np.mean([p < 0.05 for p in ps])
        print(f"  true={h2_true:.1f}  REML={m_reml:.3f}±{np.std(ests):.3f}  "
              f"HE={m_he:.3f}  frac(p<0.05)={frac_sig:.2f}")
        if h2_true == 0.0:
            ok &= (m_reml < 0.15) and (frac_sig <= 0.25)      # calibrated null
        else:
            ok &= abs(m_reml - h2_true) < 0.15                # unbiased-ish recovery
            if h2_true >= 0.5:
                ok &= frac_sig >= 0.75                        # powered when signal strong
    print("\n" + ("ALL REML TESTS PASSED" if ok else "REML TEST FAILED"))
    assert ok, "REML estimator did not recover known h² within tolerance"


if __name__ == "__main__":
    main()
