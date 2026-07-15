#!/usr/bin/env python
"""Phase 9 — direction-of-effect accuracy: does Enformer predict the correct SIGN of eQTL effects?

Documented in docs/PHASE9_direction_of_effect.md. The Sasse 2023 diagnostic and the mechanistic explanation
for why Enformer zero-shot ρ≈0: for measured cis-eQTLs (significant lead variants on EUR), compare the SIGN of
Enformer's ISM CAGE effect (alt allele) to the SIGN of the measured eQTL effect. Sign-agreement significantly
above 50% would mean the model captures direction; ~50% reproduces the documented wrong-direction failure.

Also a scatter of measured eQTL effect size vs Enformer ISM delta. Light GPU (~1-2 min: ~K forwards/gene).
Run: .venv/bin/python scripts/phase9_direction_of_effect.py
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, binomtest

from paths import REF_DIR, PROCESSED, RESULTS, CACHE, kgp_vcf_url
from genome import Reference
from personal import stream_window_genotypes, write_sample_file
import enformer_ism
import backbone

LENGTH = 196_608
MAF_MIN = 0.05
TOP_K = 5           # strongest eQTLs per gene
MIN_ABS_R = 0.15    # |Pearson r| threshold (~p<0.005 at n≈360) → real eQTLs


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    eur = [s for s in im.index if im.loc[s, "Superpopulation"] == "EUR"]
    sf = write_sample_file(eur, CACHE / "p9_eur.txt")
    url = kgp_vcf_url("chr22")
    backbone.load_enformer()
    half = LENGTH // 2

    rows = []; t0 = time.time()
    for gid in selected:
        g = gm.loc[gid]
        try:
            wg = stream_window_genotypes(ref, url, "chr22", int(g.tss), eur, sf, length=LENGTH, strand="+")
        except Exception:
            continue
        if wg.n_variants == 0:
            continue
        gt = wg.gt.astype(np.float32); gt[gt < 0] = np.nan
        D = np.nansum(gt, axis=2); af = np.nanmean(D, axis=1) / 2.0
        keep = (af >= MAF_MIN) & (af <= 1 - MAF_MIN)
        idx = np.where(keep)[0]
        if len(idx) < 5:
            continue
        y = target.loc[gid, eur].to_numpy(float)
        # measured eQTL effect (marginal r of alt-dosage with expression) per common SNP
        rs = np.zeros(len(idx))
        for k, i in enumerate(idx):
            d = D[i].copy(); m = np.nanmean(d); d[np.isnan(d)] = m
            rs[k] = pearsonr(d, y).statistic if np.std(d) > 1e-9 else 0.0
        sigk = idx[np.abs(rs) >= MIN_ABS_R]
        sigr = rs[np.abs(rs) >= MIN_ABS_R]
        if len(sigk) == 0:
            continue
        order = np.argsort(np.abs(sigr))[::-1][:TOP_K]
        sel_i = sigk[order]; sel_r = sigr[order]
        # Enformer ISM deltas (alt allele) for the selected eQTLs
        deltas, _ = enformer_ism.variant_cage_deltas(ref, "chr22", int(g.tss), g.strand,
                                                     wg.positions[sel_i], wg.alt_alleles[sel_i], batch_size=8)
        for r_meas, dlt in zip(sel_r, deltas):
            rows.append(dict(gene=g.gene_name, gene_id=gid, measured_r=float(r_meas),
                             enformer_ism=float(dlt)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase9_direction.csv", index=False)
    print(f"eQTLs tested: {len(df)} across {df.gene_id.nunique()} genes ({time.time()-t0:.0f}s)")

    # ---- sign-agreement (only where Enformer predicts a nonzero effect) ---------
    nz = df[df.enformer_ism.abs() > 1e-6]
    agree = (np.sign(nz.measured_r) == np.sign(nz.enformer_ism))
    frac = float(agree.mean()) if len(nz) else float("nan")
    p = binomtest(int(agree.sum()), len(nz), 0.5).pvalue if len(nz) else float("nan")
    print(f"\nDirection-of-effect sign-agreement (Enformer ISM vs measured eQTL):")
    print(f"  all eQTLs with nonzero Enformer effect: {agree.sum()}/{len(nz)} = {frac:.1%}  (vs 50% null, binomial p={p:.2f})")
    # high-confidence subset: top-quartile |ISM delta|
    if len(nz):
        thr = nz.enformer_ism.abs().quantile(0.75)
        hc = nz[nz.enformer_ism.abs() >= thr]
        a2 = (np.sign(hc.measured_r) == np.sign(hc.enformer_ism))
        p2 = binomtest(int(a2.sum()), len(hc), 0.5).pvalue
        print(f"  high-confidence (top-quartile |ISM|):   {a2.sum()}/{len(hc)} = {a2.mean():.1%}  (binomial p={p2:.2f})")
    # correlation of magnitudes
    if len(nz) > 2:
        rr = pearsonr(nz.measured_r, nz.enformer_ism)
        print(f"  corr(measured eQTL r, Enformer ISM delta) = {rr.statistic:+.3f} (p={rr.pvalue:.2g})")

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.axhline(0, color="k", lw=0.6, ls=":"); ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.scatter(df.measured_r, df.enformer_ism, s=20, alpha=0.6, c="#4C72B0")
    ax.set_xlabel("measured eQTL effect (Pearson r, alt allele vs expression)")
    ax.set_ylabel("Enformer ISM effect (CAGE Δ, alt allele)")
    ax.set_title(f"Direction of effect: Enformer vs measured eQTLs\n"
                 f"sign-agreement {frac:.0%} (n={len(nz)}, null 50%, p={p:.2f}) — Q1/Q3 quadrants = agree")
    fig.tight_layout(); fig.savefig(RESULTS / "phase9_direction.png", dpi=130)
    print("\nwrote results/phase9_direction.csv, phase9_direction.png")
    print("Phase 9 complete.")


if __name__ == "__main__":
    main()
