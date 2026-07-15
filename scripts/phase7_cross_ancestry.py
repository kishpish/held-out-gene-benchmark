#!/usr/bin/env python
"""Phase 7 — cross-ancestry portability of cross-individual prediction (METHODOLOGY §4.3, §8.2).

Documented in docs/PHASE7_cross_ancestry.md. Trains the linear cross-individual models on EUR individuals and
evaluates them (a) within-ancestry on held-out EUR individuals and (b) cross-ancestry on AFR (YRI) individuals,
on the learnable chr22 genes. Quantifies the portability drop — the well-documented failure of PRS/TWAS-style
models to transfer across ancestries — as a distinctive axis of the benchmark. CPU only.

Run: .venv/bin/python scripts/phase7_cross_ancestry.py
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
from scipy.stats import spearmanr, pearsonr, wilcoxon
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from paths import PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
from baselines import cis_dosage
from personal import write_sample_file

WORKERS = 8


def _rho(p, y):
    return 0.0 if np.std(p) < 1e-9 else float(spearmanr(p, y).statistic)


def eval_gene(gid, gm, target, url, samples, sf, tr, eur_te, afr):
    g = gm.loc[gid]
    X, af, pos = cis_dosage(url, "chr22", int(g.tss), samples, sf)
    if X.shape[1] == 0:
        return None
    y = target.loc[gid, samples].to_numpy(float)
    Xtr, ytr = X[tr], y[tr]
    out = dict(gene_id=gid, gene=g.gene_name, n_snp=X.shape[1])
    # lead-eQTL: best SNP on EUR-train
    c = np.array([abs(pearsonr(Xtr[:, j], ytr).statistic) if np.std(Xtr[:, j]) > 1e-9 else 0.0
                  for j in range(X.shape[1])])
    j = int(np.argmax(c)); b = np.polyfit(Xtr[:, j], ytr, 1)
    out["lead_within"] = _rho(np.polyval(b, X[eur_te][:, j]), y[eur_te])
    out["lead_cross"] = _rho(np.polyval(b, X[afr][:, j]), y[afr])
    # elastic-net on EUR-train
    sc = StandardScaler().fit(Xtr)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-3,
                    selection="random", random_state=0).fit(sc.transform(Xtr), ytr)
    out["enet_within"] = _rho(en.predict(sc.transform(X[eur_te])), y[eur_te])
    out["enet_cross"] = _rho(en.predict(sc.transform(X[afr])), y[afr])
    return out


def main() -> None:
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    isplit = json.loads((SPLITS / "individual_split.json").read_text())
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()

    eur = set(im.index[im.Superpopulation == "EUR"])
    afr_ids = list(im.index[im.Superpopulation == "AFR"])
    eur_train = [s for s in isplit["train"] if s in eur]
    eur_test = [s for s in isplit["test"] if s in eur]
    samples = eur_train + eur_test + afr_ids
    tr = np.arange(len(eur_train))
    eur_te = np.arange(len(eur_train), len(eur_train) + len(eur_test))
    afr = np.arange(len(eur_train) + len(eur_test), len(samples))
    sf = write_sample_file(samples, CACHE / "p7_samples.txt")
    url = kgp_vcf_url("chr22")
    print(f"train EUR={len(eur_train)} | within-test EUR={len(eur_test)} | cross-test AFR={len(afr_ids)} "
          f"| genes={len(selected)}")

    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(eval_gene, gid, gm, target, url, samples, sf, tr, eur_te, afr)
                for gid in selected]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase7_cross_ancestry.csv", index=False)
    print(f"evaluated {len(df)} genes ({time.time()-t0:.0f}s)")

    def med_ci(v, n=2000):
        rng = np.random.default_rng(0); v = np.asarray(v)
        m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
        return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

    print("\nMedian per-gene cross-individual ρ [95% CI] — within-EUR vs cross-ancestry (AFR):")
    for meth in ["lead", "enet"]:
        w = med_ci(df[f"{meth}_within"]); c = med_ci(df[f"{meth}_cross"])
        drop = (w[0] - c[0]) / w[0] * 100 if w[0] > 0 else float("nan")
        p = wilcoxon(df[f"{meth}_within"], df[f"{meth}_cross"], alternative="greater").pvalue
        name = "lead-eQTL " if meth == "lead" else "elastic-net"
        print(f"  {name}: within={w[0]:+.3f} [{w[1]:+.3f},{w[2]:+.3f}]  "
              f"cross={c[0]:+.3f} [{c[1]:+.3f},{c[2]:+.3f}]  drop={drop:.0f}%  (within>cross p={p:.1e})")

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    series = [("lead-eQTL\nwithin-EUR", df.lead_within, "#DD8452"),
              ("lead-eQTL\ncross-AFR", df.lead_cross, "#F0C0A0"),
              ("elastic-net\nwithin-EUR", df.enet_within, "#4C72B0"),
              ("elastic-net\ncross-AFR", df.enet_cross, "#A8C0E0")]
    bp = ax.boxplot([s[1] for s in series], tick_labels=[s[0] for s in series],
                    showmeans=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], series):
        patch.set_facecolor(s[2]); patch.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("per-gene cross-individual Spearman ρ")
    ax.set_title(f"Cross-ancestry portability (chr22, n={len(df)} learnable genes)\n"
                 "EUR-trained linear models tested within-EUR vs on AFR individuals")
    fig.tight_layout(); fig.savefig(RESULTS / "phase7_cross_ancestry.png", dpi=130)
    print("\nwrote results/phase7_cross_ancestry.csv, phase7_cross_ancestry.png")
    print("Phase 7 cross-ancestry complete.")


if __name__ == "__main__":
    main()
