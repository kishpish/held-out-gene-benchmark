#!/usr/bin/env python
"""Phase 10b — sub-population-corrected within-EUR baselines (isolate pure cis signal).

Phase 10's ancestry-only control revealed that, within Europeans on the light (ancestry-retained) target,
EUR sub-population structure (CEU/FIN/GBR/TSI) alone reaches ρ≈0.20 — so the within-EUR linear signal is
confounded by sub-structure. This script residualizes the target on sub-population labels (fit on TRAIN
individuals only) and re-runs the linear baselines + controls, giving the honest PURE-cis cross-individual ρ.

Compares uncorrected vs sub-pop-corrected; the ancestry-only control on the corrected target must drop to ~0.
CPU only. Run: .venv/bin/python scripts/phase10b_ancestry_corrected.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

from paths import PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
from baselines import cis_dosage
from personal import write_sample_file

WORKERS = 8


def _rho(p, y):
    return 0.0 if np.std(p) < 1e-9 else float(spearmanr(p, y).statistic)


def _enet(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-3,
                    selection="random", random_state=0).fit(sc.transform(Xtr), ytr)
    return en.predict(sc.transform(Xte))


def _lead(Xtr, ytr, Xte):
    c = np.array([abs(pearsonr(Xtr[:, j], ytr).statistic) if np.std(Xtr[:, j]) > 1e-9 else 0.0
                  for j in range(Xtr.shape[1])])
    j = int(np.argmax(c)); b = np.polyfit(Xtr[:, j], ytr, 1)
    return np.polyval(b, Xte[:, j])


def eval_gene(gid, gm, target, P, url, samples, sf, tr, te):
    g = gm.loc[gid]
    X, af, pos = cis_dosage(url, "chr22", int(g.tss), samples, sf)
    if X.shape[1] == 0:
        return None
    y = target.loc[gid, samples].to_numpy(float)
    # sub-population correction: residualize target on sub-pop one-hot, fit on TRAIN only
    lr = LinearRegression().fit(P[tr], y[tr])
    yc = y - lr.predict(P)                        # corrected target (pure of sub-pop means)
    out = dict(gene_id=gid, gene=g.gene_name)
    # uncorrected
    out["enet_uncorr"] = _rho(_enet(X[tr], y[tr], X[te]), y[te])
    out["lead_uncorr"] = _rho(_lead(X[tr], y[tr], X[te]), y[te])
    out["anc_uncorr"]  = _rho(LinearRegression().fit(P[tr], y[tr]).predict(P[te]), y[te])
    # corrected
    out["enet_corr"] = _rho(_enet(X[tr], yc[tr], X[te]), yc[te])
    out["lead_corr"] = _rho(_lead(X[tr], yc[tr], X[te]), yc[te])
    out["anc_corr"]  = _rho(LinearRegression().fit(P[tr], yc[tr]).predict(P[te]), yc[te])
    return out


def main() -> None:
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    isplit = json.loads((SPLITS / "individual_split.json").read_text())
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()

    eur = set(im.index[im.Superpopulation == "EUR"])
    train = [s for s in isplit["train"] if s in eur]
    test = [s for s in isplit["test"] if s in eur]
    samples = train + test
    tr = np.arange(len(train)); te = np.arange(len(train), len(samples))
    P = pd.get_dummies(im.loc[samples, "Population"].astype("category")).to_numpy(float)
    sf = write_sample_file(samples, CACHE / "p10b_eur.txt")
    url = kgp_vcf_url("chr22")
    print(f"sub-pop-corrected baselines on {len(selected)} learnable genes | EUR train={len(train)} test={len(test)}")

    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(eval_gene, gid, gm, target, P, url, samples, sf, tr, te)
                               for gid in selected]):
            r = f.result()
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase10b_ancestry_corrected.csv", index=False)
    print(f"evaluated {len(df)} genes ({time.time()-t0:.0f}s)")

    def med_ci(v, n=2000):
        rng = np.random.default_rng(0); v = np.asarray(v)
        m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
        return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

    print("\nMedian per-gene cross-individual ρ [95% CI] — uncorrected vs sub-pop-corrected:")
    for label, cols in [("elastic-net", ("enet_uncorr", "enet_corr")),
                        ("lead-eQTL", ("lead_uncorr", "lead_corr")),
                        ("ancestry-only ctrl", ("anc_uncorr", "anc_corr"))]:
        u = med_ci(df[cols[0]]); c = med_ci(df[cols[1]])
        print(f"  {label:20s} uncorrected={u[0]:+.3f} [{u[1]:+.3f},{u[2]:+.3f}]   "
              f"corrected={c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]")

    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    groups = [("elastic-net", "enet_uncorr", "enet_corr"),
              ("lead-eQTL", "lead_uncorr", "lead_corr"),
              ("ancestry-only", "anc_uncorr", "anc_corr")]
    pos = np.arange(len(groups))
    bu = ax.boxplot([df[g[1]] for g in groups], positions=pos - 0.18, widths=0.32, patch_artist=True, showmeans=True)
    bc = ax.boxplot([df[g[2]] for g in groups], positions=pos + 0.18, widths=0.32, patch_artist=True, showmeans=True)
    for b, col in [(bu, "#BBBBBB"), (bc, "#4C72B0")]:
        for p in b["boxes"]:
            p.set_facecolor(col); p.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(pos); ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("per-gene cross-individual ρ (held-out EUR)")
    ax.set_title("Sub-population correction isolates pure-cis signal\n(grey = uncorrected, blue = sub-pop-corrected)")
    fig.tight_layout(); fig.savefig(RESULTS / "phase10b_ancestry_corrected.png", dpi=130)
    print("\nwrote results/phase10b_ancestry_corrected.csv, phase10b_ancestry_corrected.png")
    print("Phase 10b complete.")


if __name__ == "__main__":
    main()
