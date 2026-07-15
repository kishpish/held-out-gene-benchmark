#!/usr/bin/env python
"""Phase 6c — does a richer Enformer-based variant feature transfer where motif deltas didn't? (§6.3)

Documented in docs/PHASE6c_enformer_prior.md. Augments the Phase-6 gene-agnostic transfer model with a
per-variant **Enformer ISM effect** (CAGE delta of the single alt allele vs reference; individual-independent
→ O(genes×variants)). Trains three heads on TRAIN-chromosome genes — motif-only, Enformer-only, and combined —
and evaluates each ZERO-SHOT on held-out chr22 genes. Isolates whether Enformer's learned representation
carries transferable directional grammar the 10 hand-picked motifs miss.

Scoped GPU job (~15-20 min): capped genes and variants keep the ISM forward count tractable.
Run: .venv/bin/python scripts/phase6c_enformer_prior.py
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from paths import REF_DIR, PROCESSED, RESULTS, SPLITS, CACHE, RAW, kgp_vcf_url
from genome import Reference, norm_chrom
from personal import stream_window_genotypes, write_sample_file
import motifs as motif_mod
import transfer_model as tm
import enformer_ism
import backbone
import json

LENGTH = 196_608
MAF_MIN = 0.05
MAX_VAR = 40           # variants/gene (caps ISM forwards)
N_TRAIN = 45           # train-chromosome genes
SEED = 0


def gene_matrix(ref, chrom, tss, strand, samples, sf, motif_list, want_ism):
    wg = stream_window_genotypes(ref, kgp_vcf_url(chrom), chrom, tss, samples, sf,
                                 length=LENGTH, strand="+")
    if wg.n_variants == 0:
        return None
    gt = wg.gt.astype(np.float32); gt[gt < 0] = np.nan
    D = np.nansum(gt, axis=2); af = np.nanmean(D, axis=1) / 2.0
    keep = (af >= MAF_MIN) & (af <= 1 - MAF_MIN)
    idx = np.where(keep)[0]
    if len(idx) < 5:
        return None
    het = 2 * af[idx] * (1 - af[idx])
    idx = idx[np.argsort(het)[::-1][:MAX_VAR]]
    pos = wg.positions[idx]; alt = wg.alt_alleles[idx]; refa = wg.ref_alleles[idx]
    feats, dpos = [], []
    for j, i in enumerate(idx):
        f = tm.variant_features_win(wg.ref_window, wg.window_start0, int(wg.positions[i]),
                                    wg.ref_alleles[i], wg.alt_alleles[i], tss, strand, motif_list, float(af[i]))
        if f is None:
            dpos.append(None); continue
        dpos.append(f)
    good = [k for k, f in enumerate(dpos) if f is not None]
    motif_feats = np.array([dpos[k] for k in good], np.float32)
    Dg = D[idx][good]; cm = np.nanmean(Dg, axis=1, keepdims=True)
    nanx = np.where(np.isnan(Dg)); Dg[nanx] = np.take(cm, nanx[0])
    ism = None
    if want_ism:
        deltas, _ = enformer_ism.variant_cage_deltas(ref, chrom, tss, strand,
                                                     pos[good], alt[good], batch_size=8)
        ism = deltas.reshape(-1, 1).astype(np.float32)
    return motif_feats, Dg, ism


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    motif_list = motif_mod.load_motifs(RAW / "jaspar")
    h2 = pd.read_csv(RESULTS / "phase3_cis_h2_chr22.csv").set_index("gene_id")
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    isplit = json.loads((SPLITS / "individual_split.json").read_text())
    eur = set(im.index[im.Superpopulation == "EUR"])
    tr_ind = [s for s in isplit["train"] if s in eur]
    te_ind = [s for s in isplit["test"] if s in eur]
    sf_tr = write_sample_file(tr_ind, CACHE / "p6c_tr.txt")
    sf_te = write_sample_file(te_ind, CACHE / "p6c_te.txt")
    backbone.load_enformer()

    half = LENGTH // 2
    train_chroms = [f"chr{i}" for i in range(1, 19)]
    clen = {c: len(ref._fa[norm_chrom(c, ref.style)]) for c in train_chroms}
    pool = gm[gm.chrom.isin(train_chroms) & gm.index.isin(target.index)]
    pool = pool[pool.apply(lambda r: half < r.tss < clen[r.chrom] - half, axis=1)]
    tg = pool.sample(N_TRAIN, random_state=SEED)
    test_genes = list(h2.loc[[g for g in selected if g in h2.index]]
                      .sort_values("h2_eur", ascending=False).head(40).index)
    print(f"train genes {len(tg)} | test genes {len(test_genes)} | "
          f"≤{MAX_VAR} var/gene → ~{(len(tg)+len(test_genes))*(MAX_VAR+1)} ISM forwards")

    # ---- assemble training data (motif + ISM features, marginal-β labels) -------
    t0 = time.time()
    Xm, Xi, yb = [], [], []
    for gid in tg.index:
        g = gm.loc[gid]
        out = gene_matrix(ref, g.chrom, int(g.tss), g.strand, tr_ind, sf_tr, motif_list, want_ism=True)
        if out is None:
            continue
        mf, D, ism = out
        y = target.loc[gid, tr_ind].to_numpy(float)
        Dc = D - D.mean(axis=1, keepdims=True); yc = y - y.mean()
        denom = (Dc * Dc).sum(axis=1); denom[denom < 1e-8] = np.nan
        beta = (Dc * yc[None, :]).sum(axis=1) / denom
        ok = np.isfinite(beta)
        Xm.append(mf[ok]); Xi.append(ism[ok]); yb.append(beta[ok])
    Xm = np.vstack(Xm); Xi = np.vstack(Xi); yb = np.concatenate(yb)
    print(f"training variants: {len(yb)} ({time.time()-t0:.0f}s)")

    Xc = np.hstack([Xm, Xi])
    train_X = {"motif_only": Xm, "enformer_only": Xi, "motif+enformer": Xc}
    heads = {name: tm.TransferModel.train(X, yb) for name, X in train_X.items()}
    for name, h in heads.items():
        print(f"  head {name:16s} train R²={h.model.score(train_X[name], yb):.3f}")

    # ---- zero-shot eval on held-out chr22 genes --------------------------------
    rows = []
    for gid in test_genes:
        g = gm.loc[gid]
        out = gene_matrix(ref, "chr22", int(g.tss), g.strand, te_ind, sf_te, motif_list, want_ism=True)
        if out is None:
            continue
        mf, D, ism = out
        y = target.loc[gid, te_ind].to_numpy(float)
        preds = {
            "motif_only": heads["motif_only"].predict_effects(mf),
            "enformer_only": heads["enformer_only"].predict_effects(ism),
            "motif+enformer": heads["motif+enformer"].predict_effects(np.hstack([mf, ism])),
        }
        row = dict(gene_id=gid, gene=g.gene_name, n_var=len(mf))
        for name, bh in preds.items():
            score = tm.aggregate_individual_scores(D, bh)
            row[f"rho_{name}"] = 0.0 if np.std(score) < 1e-9 else float(spearmanr(score, y).statistic)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase6c_enformer_prior.csv", index=False)
    print(f"\nzero-shot on {len(df)} held-out genes ({time.time()-t0:.0f}s total):")
    for name in ["motif_only", "enformer_only", "motif+enformer"]:
        v = df[f"rho_{name}"]
        try:
            p = wilcoxon(v, alternative="greater").pvalue
        except Exception:
            p = np.nan
        print(f"  {name:16s} median ρ={v.median():+.3f}  frac(ρ>0)={float((v>0).mean()):.2f}  p={p:.3f}")

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    order = ["motif_only", "enformer_only", "motif+enformer"]
    labels = ["motif deltas\n(Phase 6 v1)", "Enformer ISM\n(§6.3)", "motif + Enformer"]
    bp = ax.boxplot([df[f"rho_{n}"] for n in order], tick_labels=labels, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#55A868", "#4C72B0", "#8172B3"]):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("zero-shot held-out-gene cross-individual ρ")
    ax.set_title("Does an Enformer-based variant feature transfer?\n(gene-agnostic head, held-out chr22 genes)")
    fig.tight_layout(); fig.savefig(RESULTS / "phase6c_enformer_prior.png", dpi=130)
    print("\nwrote results/phase6c_enformer_prior.csv, phase6c_enformer_prior.png")
    print("Phase 6c complete.")


if __name__ == "__main__":
    main()
