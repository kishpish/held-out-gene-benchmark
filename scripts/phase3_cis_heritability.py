#!/usr/bin/env python
"""Phase 3 — estimate per-gene cis-h² (correlation ceiling) + select evaluation genes.

Documented in docs/PHASE3_cis_heritability.md. For each chr22 gene: stream the 196,608 bp window
genotypes (reference-free), build the standardized-genotype GRM, and estimate h²_cis by REML on the
centered (INT) target — primary within the homogeneous EUR subset, secondary on the full cohort.
Produces results/phase3_cis_h2_chr22.csv + results/phase3_cis_h2_distribution.png and the selected
"learnable" gene set.

chr22 validates the pipeline (no reference needed → genome-wide runs identically). GPU not used.
Run: .venv/bin/python scripts/phase3_cis_heritability.py
"""
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PROCESSED, RESULTS, CACHE, kgp_vcf_url
import personal
from cis_heritability import build_grm, reml_h2, he_regression_h2

LENGTH = 196_608
MAF_MIN = 0.01
WORKERS = 8


def estimate_gene(args):
    gid, chrom, tss, strand, y_eur, y_all, url, sf_eur, sf_all, eur_idx = args
    half = LENGTH // 2
    start1, end1 = tss - half, tss + half - 1
    try:
        # EUR (primary)
        _, D, _ = personal.stream_dosage_matrix(url, chrom, start1, end1, eur_idx, sf_eur)
        K, m = build_grm(D, MAF_MIN)
        r = reml_h2(y_eur, K)
        he = he_regression_h2(y_eur, K)
        return dict(gene_id=gid, chrom=chrom, tss=tss, strand=strand, n_snp=m,
                    h2_eur=r.h2, se_eur=r.se, p_eur=r.pval, he_eur=he)
    except Exception as e:
        return dict(gene_id=gid, chrom=chrom, tss=tss, strand=strand, n_snp=0,
                    h2_eur=np.nan, se_eur=np.nan, p_eur=np.nan, he_eur=np.nan, error=str(e)[:80])


def main() -> None:
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")

    eur = list(im.index[im.Superpopulation == "EUR"])
    url = kgp_vcf_url("chr22")
    sf_eur = personal.write_sample_file(eur, CACHE / "eur_samples.txt")
    sf_all = personal.write_sample_file(list(im.index), CACHE / "all_samples.txt")

    half = LENGTH // 2
    chr22 = gm[(gm.chrom == "chr22") & (gm.tss > half)].copy()
    chr22 = chr22[chr22.index.isin(target.index)]
    print(f"chr22 genes to estimate: {len(chr22)} | EUR individuals: {len(eur)}")

    jobs = []
    for gid, g in chr22.iterrows():
        y_eur = target.loc[gid, eur].to_numpy(float)
        jobs.append((gid, "chr22", int(g.tss), g.strand, y_eur, None, url, sf_eur, sf_all, eur))

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(estimate_gene, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            rows.append(f.result())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)} genes ({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows).set_index("gene_id")
    df = df.merge(gm[["gene_name"]], left_index=True, right_index=True, how="left")
    df.to_csv(RESULTS / "phase3_cis_h2_chr22.csv")
    ok = df.dropna(subset=["h2_eur"])
    print(f"\nestimated {len(ok)}/{len(df)} genes in {time.time()-t0:.0f}s")

    # ---- gene selection + summary ----------------------------------------------
    print("\ncis-h² (EUR, 196kb window) summary:")
    print(f"  median h²={ok.h2_eur.median():.3f}  mean={ok.h2_eur.mean():.3f}  "
          f"90th pct={ok.h2_eur.quantile(0.9):.3f}")
    for thr in [0.05, 0.10, 0.20]:
        n = int(((ok.h2_eur > thr) & (ok.p_eur < 0.05)).sum())
        print(f"  learnable genes (h²>{thr:.2f} & p<0.05): {n} "
              f"({100*n/len(ok):.0f}%)  → √h² ceiling ≥ {np.sqrt(thr):.2f}")
    sel = ok[(ok.h2_eur > 0.05) & (ok.p_eur < 0.05)]
    sel_ids = list(sel.index)
    (RESULTS / "phase3_selected_genes_chr22.txt").write_text("\n".join(sel_ids) + "\n")
    print(f"  → selected evaluation gene set (default h²>0.05 & p<0.05): {len(sel_ids)} genes")

    # ---- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(ok.h2_eur.clip(0, 1), bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
    axes[0].axvline(0.05, color="#C44E52", ls="--", lw=1.2, label="selection thr 0.05")
    axes[0].set_xlabel("cis-h² (EUR, 196,608 bp window)"); axes[0].set_ylabel("genes")
    axes[0].set_title(f"Per-gene cis-heritability (chr22, n={len(ok)} genes)")
    axes[0].legend(frameon=False, fontsize=9)
    # ceiling curve: #genes retained vs threshold
    thrs = np.linspace(0, 0.6, 60)
    retained = [(ok.h2_eur > t).sum() for t in thrs]
    axes[1].plot(thrs, retained, color="#55A868", lw=2)
    axes[1].axvline(0.05, color="#C44E52", ls="--", lw=1.2)
    axes[1].set_xlabel("cis-h² threshold"); axes[1].set_ylabel("genes retained")
    axes[1].set_title("Learnable-gene set vs threshold\n(√h² = per-gene correlation ceiling)")
    for t in [0.05, 0.2]:
        axes[1].annotate(f"√{t}={np.sqrt(t):.2f}", (t, (ok.h2_eur > t).sum()),
                         fontsize=8, xytext=(t+0.03, (ok.h2_eur > t).sum()+5))
    fig.tight_layout(); fig.savefig(RESULTS / "phase3_cis_h2_distribution.png", dpi=130)
    print("\nwrote results/phase3_cis_h2_chr22.csv, phase3_selected_genes_chr22.txt, phase3_cis_h2_distribution.png")
    print("Phase 3 cis-h² complete.")


if __name__ == "__main__":
    main()
