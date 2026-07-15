#!/usr/bin/env python
"""Phase 8a — cross-chromosome robustness of the reproduction cell (METHODOLOGY §10).

Documented in docs/PHASE8_robustness.md. Re-runs, on multiple independent chromosomes, the cis-h² gene
selection AND the within-EUR linear baselines from a SINGLE genotype stream per gene (efficient). Shows the
learnable-gene fraction and the linear cross-individual signal are consistent genome-wide, not chr22-specific
— the key robustness check for the benchmark. CPU only.

Run: .venv/bin/python scripts/phase8a_multichrom_robustness.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from paths import PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
from genome import Reference, norm_chrom
from paths import REF_DIR
from personal import stream_window_genotypes, write_sample_file
from cis_heritability import build_grm, reml_h2

CHROMS = ["chr22", "chr5", "chr11", "chr16"]
N_PER_CHROM = 180
LENGTH = 196_608
MAF_MIN = 0.05
WORKERS = 8
SEED = 0


def _rho(p, y):
    return 0.0 if np.std(p) < 1e-9 else float(spearmanr(p, y).statistic)


_tls = threading.local()


def _ref():
    """Per-thread Reference (pyfaidx file handles are not safe to share across threads)."""
    r = getattr(_tls, "ref", None)
    if r is None:
        r = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
        _tls.ref = r
    return r


def eval_gene(gid, chrom, tss, gm, target, sf, eur, tr_pos, te_pos):
    """One stream → (cis-h² on EUR) + (within-EUR lead-eQTL & elastic-net train→test)."""
    try:
        wg = stream_window_genotypes(_ref(), kgp_vcf_url(chrom), chrom, tss, eur, sf,
                                     length=LENGTH, strand="+")
    except Exception:
        return None
    if wg.n_variants == 0:
        return None
    gt = wg.gt.astype(np.float32); gt[gt < 0] = np.nan
    D = np.nansum(gt, axis=2); af = np.nanmean(D, axis=1) / 2.0
    keep = (af >= MAF_MIN) & (af <= 1 - MAF_MIN)
    D = D[keep]
    if D.shape[0] < 5:
        return None
    cm = np.nanmean(D, axis=1, keepdims=True); nanx = np.where(np.isnan(D)); D[nanx] = np.take(cm, nanx[0])
    y = target.loc[gid, eur].to_numpy(float)
    # cis-h² on all EUR
    K, m = build_grm(D, MAF_MIN)
    h2 = reml_h2(y, K)
    # linear within-EUR train->test
    X = D.T
    Xtr, ytr = X[tr_pos], y[tr_pos]; Xte, yte = X[te_pos], y[te_pos]
    c = np.array([abs(pearsonr(Xtr[:, j], ytr).statistic) if np.std(Xtr[:, j]) > 1e-9 else 0.0
                  for j in range(X.shape[1])])
    j = int(np.argmax(c)); b = np.polyfit(Xtr[:, j], ytr, 1)
    rho_lead = _rho(np.polyval(b, Xte[:, j]), yte)
    sc = StandardScaler().fit(Xtr)
    en = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=1000, tol=1e-3,
                    selection="random", random_state=0).fit(sc.transform(Xtr), ytr)
    rho_enet = _rho(en.predict(sc.transform(Xte)), yte)
    return dict(gene_id=gid, chrom=chrom, h2=h2.h2, p_h2=h2.pval,
                rho_lead=rho_lead, rho_enet=rho_enet, n_snp=X.shape[1])


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    isplit = json.loads((SPLITS / "individual_split.json").read_text())
    eurset = set(im.index[im.Superpopulation == "EUR"])
    eur = [s for s in im.index if s in eurset]                 # all EUR (fixed order)
    tr = set(isplit["train"]) & eurset; te = set(isplit["test"]) & eurset
    tr_pos = np.array([i for i, s in enumerate(eur) if s in tr])
    te_pos = np.array([i for i, s in enumerate(eur) if s in te])
    sf = write_sample_file(eur, CACHE / "p8_eur.txt")
    half = LENGTH // 2
    print(f"EUR total={len(eur)} (train {len(tr_pos)}, test {len(te_pos)}) | chroms {CHROMS}")

    jobs = []
    for c in CHROMS:
        clen = len(ref._fa[norm_chrom(c, ref.style)])
        pool = gm[(gm.chrom == c) & gm.index.isin(target.index)]
        pool = pool[(pool.tss > half) & (pool.tss < clen - half)]
        pick = pool.sample(min(N_PER_CHROM, len(pool)), random_state=SEED)
        for gid, g in pick.iterrows():
            jobs.append((gid, c, int(g.tss)))
    print(f"total genes to evaluate: {len(jobs)}")

    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(eval_gene, gid, c, tss, gm, target, sf, eur, tr_pos, te_pos)
                for gid, c, tss in jobs]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if r:
                rows.append(r)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(jobs)} ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase8a_multichrom.csv", index=False)

    print("\nPer-chromosome robustness (learnable = h²>0.05 & p<0.05):")
    summ = []
    for c in CHROMS:
        d = df[df.chrom == c]; learn = d[(d.h2 > 0.05) & (d.p_h2 < 0.05)]
        row = dict(chrom=c, n=len(d), frac_learnable=len(learn) / max(1, len(d)),
                   median_h2=float(d.h2.median()),
                   lead_learn=float(learn.rho_lead.median()) if len(learn) else np.nan,
                   enet_learn=float(learn.rho_enet.median()) if len(learn) else np.nan)
        summ.append(row)
        print(f"  {c}: n={row['n']:3d}  learnable={row['frac_learnable']*100:.0f}%  "
              f"median h²={row['median_h2']:.3f}  lead ρ={row['lead_learn']:+.3f}  enet ρ={row['enet_learn']:+.3f}")
    sdf = pd.DataFrame(summ); sdf.to_csv(RESULTS / "phase8a_summary.csv", index=False)

    # ---- figure: per-chromosome linear ρ on learnable genes -------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    data_lead = [df[(df.chrom == c) & (df.h2 > 0.05) & (df.p_h2 < 0.05)].rho_lead for c in CHROMS]
    data_enet = [df[(df.chrom == c) & (df.h2 > 0.05) & (df.p_h2 < 0.05)].rho_enet for c in CHROMS]
    pos = np.arange(len(CHROMS))
    b1 = ax.boxplot(data_lead, positions=pos - 0.18, widths=0.32, patch_artist=True, showmeans=True)
    b2 = ax.boxplot(data_enet, positions=pos + 0.18, widths=0.32, patch_artist=True, showmeans=True)
    for b, col in [(b1, "#DD8452"), (b2, "#4C72B0")]:
        for p in b["boxes"]:
            p.set_facecolor(col); p.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xticks(pos); ax.set_xticklabels(CHROMS)
    ax.set_ylabel("per-gene cross-individual ρ (learnable genes, held-out EUR)")
    ax.set_title("Reproduction cell is consistent across chromosomes\n(orange = lead-eQTL, blue = elastic-net)")
    fig.tight_layout(); fig.savefig(RESULTS / "phase8a_multichrom.png", dpi=130)
    print("\nwrote results/phase8a_multichrom.csv, phase8a_summary.csv, phase8a_multichrom.png")
    print("Phase 8a complete.")


if __name__ == "__main__":
    main()
