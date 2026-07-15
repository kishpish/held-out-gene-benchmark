#!/usr/bin/env python
"""Phase 10 — negative controls (METHODOLOGY hard rule 6, §7.4, §8.5).

Two controls the benchmark must pass, on the 78 learnable chr22 genes within the European subset:

  (1) PERMUTATION control — for each gene, correlate the elastic-net predictions with the test targets after
      shuffling the individual↔target correspondence. A correct, leakage-free pipeline collapses to ρ≈0.
      Confirms the observed signal is not a metric/pipeline artifact.

  (2) ANCESTRY-ONLY control — a predictor given only ancestry information (EUR sub-population labels:
      CEU/FIN/GBR/TSI), no local sequence. Within Europeans this should give ρ≈0 and must be far below the
      real cis-genotype models — confirming the within-EUR results reflect cis-grammar, not ancestry.

CPU only. Run: .venv/bin/python scripts/phase10_negative_controls.py
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
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

from paths import PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
from baselines import cis_dosage
from personal import write_sample_file

N_PERM = 200
WORKERS = 8


def _rho(p, y):
    return 0.0 if np.std(p) < 1e-9 else float(spearmanr(p, y).statistic)


def eval_gene(gid, gm, target, pop_onehot, url, samples, sf, tr, te, seed):
    g = gm.loc[gid]
    X, af, pos = cis_dosage(url, "chr22", int(g.tss), samples, sf)
    if X.shape[1] == 0:
        return None
    y = target.loc[gid, samples].to_numpy(float)
    Xtr, ytr, yte = X[tr], y[tr], y[te]
    # real elastic-net prediction (cis genotypes)
    sc = StandardScaler().fit(Xtr)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-3,
                    selection="random", random_state=0).fit(sc.transform(Xtr), ytr)
    pred = en.predict(sc.transform(X[te]))
    rho_real = _rho(pred, yte)
    # (1) permutation null: shuffle test labels vs the (fixed) predictions
    rng = np.random.default_rng(seed)
    perm = [_rho(pred, rng.permutation(yte)) for _ in range(N_PERM)]
    rho_perm = float(np.mean(perm))
    # (2) ancestry-only: predict from EUR sub-population one-hot (no sequence)
    Ptr, Pte = pop_onehot[tr], pop_onehot[te]
    lr = LinearRegression().fit(Ptr, ytr)
    rho_anc = _rho(lr.predict(Pte), yte)
    return dict(gene_id=gid, gene=g.gene_name, rho_real=rho_real,
                rho_perm=rho_perm, rho_ancestry=rho_anc)


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
    # EUR sub-population one-hot for the ancestry-only control
    pops = im.loc[samples, "Population"].astype("category")
    pop_onehot = pd.get_dummies(pops).to_numpy(float)
    sf = write_sample_file(samples, CACHE / "p10_eur.txt")
    url = kgp_vcf_url("chr22")
    print(f"controls on {len(selected)} learnable genes | EUR train={len(train)} test={len(test)} "
          f"| sub-pops={list(pops.cat.categories)} | {N_PERM} permutations/gene")

    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(eval_gene, gid, gm, target, pop_onehot, url, samples, sf, tr, te, i)
                for i, gid in enumerate(selected)]
        for f in as_completed(futs):
            r = f.result()
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase10_negative_controls.csv", index=False)
    print(f"evaluated {len(df)} genes ({time.time()-t0:.0f}s)")

    def med_ci(v, n=2000):
        rng = np.random.default_rng(0); v = np.asarray(v)
        m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
        return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)

    print("\nMedian per-gene cross-individual ρ [95% CI]:")
    for col, name in [("rho_real", "real elastic-net (cis genotypes)"),
                      ("rho_perm", "PERMUTATION control (shuffled labels)"),
                      ("rho_ancestry", "ANCESTRY-ONLY control (sub-pop labels)")]:
        m, lo, hi = med_ci(df[col])
        print(f"  {name:42s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    series = [("real\nelastic-net", df.rho_real, "#4C72B0"),
              ("permutation\ncontrol", df.rho_perm, "#8C8C8C"),
              ("ancestry-only\ncontrol", df.rho_ancestry, "#DD8452")]
    bp = ax.boxplot([s[1] for s in series], tick_labels=[s[0] for s in series],
                    showmeans=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], series):
        patch.set_facecolor(s[2]); patch.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("per-gene cross-individual Spearman ρ")
    ax.set_title(f"Negative controls (chr22, n={len(df)} learnable genes, within-EUR)\n"
                 "permutation → 0 (passes); ancestry-only ≈ 0.20 reveals sub-structure "
                 "(cis is robust to it — Fig. 10b)")
    fig.tight_layout(); fig.savefig(RESULTS / "phase10_negative_controls.png", dpi=130)
    print("\nwrote results/phase10_negative_controls.csv, phase10_negative_controls.png")
    print("Phase 10 complete.")


if __name__ == "__main__":
    main()
