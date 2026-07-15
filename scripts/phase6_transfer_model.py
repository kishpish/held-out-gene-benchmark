#!/usr/bin/env python
"""Phase 6 — gene-agnostic transferable variant-effect model; ZERO-SHOT held-out-gene evaluation.

Documented in docs/PHASE6_transferable_variant_effect.md. Trains on TRAIN-chromosome genes (chr1-18) a map
from local motif-grammar features → per-variant marginal cis-eQTL effect, then applies it ZERO-SHOT to
held-out chr22 genes: predict each variant's effect from features, aggregate Σ dosage·β̂ per individual, and
correlate across held-out EUR individuals with the centered target. This is the exact cell where Enformer
zero-shot got ρ≈0 and per-gene linear models are (by construction) undefined without labels.

Compares: transfer model (zero-shot) vs burden baseline (zero-shot) vs Enformer zero-shot vs per-gene
elastic-net (the with-labels reference). CPU only. Run: .venv/bin/python scripts/phase6_transfer_model.py
"""
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

from paths import REF_DIR, PROCESSED, RESULTS, SPLITS, CACHE, RAW, kgp_vcf_url
from genome import Reference, norm_chrom
from personal import stream_dosage_matrix, write_sample_file
import motifs as motif_mod
import transfer_model as tm

N_TRAIN_GENES = 160
LENGTH = 196_608
MAF_MIN = 0.05
WORKERS = 8
SEED = 0


def marginal_beta(D, y):
    """Per-variant OLS slope of y on dosage. D:(n_var,n_samp) mean-imputed, y:(n_samp,)."""
    Dc = D - np.nanmean(D, axis=1, keepdims=True)
    yc = y - y.mean()
    denom = np.nansum(Dc * Dc, axis=1)
    denom[denom < 1e-8] = np.nan
    return np.nansum(Dc * yc[None, :], axis=1) / denom


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    motif_list = motif_mod.load_motifs(RAW / "jaspar")
    isplit = json.loads((SPLITS / "individual_split.json").read_text())
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    print(f"motifs: {[m.name for m in motif_list]}")

    eur = set(im.index[im.Superpopulation == "EUR"])
    tr_ind = [s for s in isplit["train"] if s in eur]
    te_ind = [s for s in isplit["test"] if s in eur]
    sf_tr = write_sample_file(tr_ind, CACHE / "p6_train.txt")
    sf_te = write_sample_file(te_ind, CACHE / "p6_test.txt")

    # need ref+alt alleles per variant → use a genotype+allele stream. Reuse personal.stream_window_genotypes
    from personal import stream_window_genotypes
    half = LENGTH // 2

    def extract(gene_id, chrom, tss, strand, samples, sf):
        wg = stream_window_genotypes(ref, kgp_vcf_url(chrom), chrom, tss, samples, sf,
                                     length=LENGTH, strand="+")   # + strand for coord math
        if wg.n_variants == 0:
            return None
        # dosage (n_var, n_samp)
        gt = wg.gt.astype(np.float32); gt[gt < 0] = np.nan
        D = np.nansum(gt, axis=2)
        af = np.nanmean(D, axis=1) / 2.0
        keep = (af >= MAF_MIN) & (af <= 1 - MAF_MIN)
        if keep.sum() == 0:
            return None
        idx = np.where(keep)[0]
        feats, dos = [], []
        for i in idx:
            f = tm.variant_features_win(wg.ref_window, wg.window_start0, int(wg.positions[i]),
                                        wg.ref_alleles[i], wg.alt_alleles[i], tss, strand, motif_list,
                                        float(af[i]))
            if f is None:
                continue
            row = D[i].copy()
            cmi = np.nanmean(row); row[np.isnan(row)] = cmi
            feats.append(f); dos.append(row)
        if not feats:
            return None
        return np.array(feats, np.float32), np.array(dos, np.float32)

    # ---- assemble training data from chr1-18 genes -----------------------------
    train_chroms = [f"chr{i}" for i in range(1, 19)]
    pool = gm[gm.chrom.isin(train_chroms) & gm.index.isin(target.index)].copy()
    # window-safe
    clen = {c: len(ref._fa[norm_chrom(c, ref.style)]) for c in train_chroms}
    pool = pool[pool.apply(lambda r: half < r.tss < clen[r.chrom] - half, axis=1)]
    tg = pool.sample(min(N_TRAIN_GENES, len(pool)), random_state=SEED)
    print(f"assembling training data from {len(tg)} train-chromosome genes...")

    def build_train(gid):
        g = gm.loc[gid]
        try:
            out = extract(gid, g.chrom, int(g.tss), g.strand, tr_ind, sf_tr)
            if out is None:
                return None
            X, D = out
            y = target.loc[gid, tr_ind].to_numpy(float)
            beta = marginal_beta(D, y)
            ok = np.isfinite(beta)
            return X[ok], beta[ok]
        except Exception:
            return None

    t0 = time.time(); Xs, ys = [], []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(build_train, gid) for gid in tg.index]):
            r = f.result()
            if r is not None:
                Xs.append(r[0]); ys.append(r[1])
    X = np.vstack(Xs); yb = np.concatenate(ys)
    print(f"training set: {X.shape[0]:,} variants × {X.shape[1]} features from {len(Xs)} genes ({time.time()-t0:.0f}s)")

    model = tm.TransferModel.train(X, yb, kind="gbm")
    print(f"train R² (variant-effect regression): {model.model.score(X, yb):.3f}")
    # persist model + training matrix so downstream (few-shot) can reuse without re-assembling
    import joblib
    joblib.dump({"model": model.model, "n_features": model.n_features},
                PROCESSED / "phase6_transfer_model.joblib")
    np.savez_compressed(PROCESSED / "phase6_train_matrix.npz", X=X, y=yb)

    # ---- ZERO-SHOT eval on held-out chr22 genes --------------------------------
    def eval_gene(gid):
        g = gm.loc[gid]
        try:
            out = extract(gid, "chr22", int(g.tss), g.strand, te_ind, sf_te)
            if out is None:
                return None
            X_g, D_g = out
            beta_hat = model.predict_effects(X_g)
            score = tm.aggregate_individual_scores(D_g, beta_hat)
            burden = tm.aggregate_individual_scores(D_g, np.ones(len(beta_hat)))
            y = target.loc[gid, te_ind].to_numpy(float)
            rho = 0.0 if np.std(score) < 1e-9 else float(spearmanr(score, y).statistic)
            rho_b = 0.0 if np.std(burden) < 1e-9 else float(spearmanr(burden, y).statistic)
            return dict(gene_id=gid, gene=g.gene_name, n_var=len(beta_hat),
                        rho_transfer=rho, rho_burden=rho_b)
        except Exception as e:
            return dict(gene_id=gid, gene=g.gene_name, error=str(e)[:60])

    t1 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(eval_gene, gid) for gid in selected]):
            rows.append(f.result())
    df = pd.DataFrame(rows)
    df = df.dropna(subset=["rho_transfer"]) if "rho_transfer" in df else df
    df.to_csv(RESULTS / "phase6_transfer_zeroshot.csv", index=False)
    print(f"zero-shot eval on {len(df)} held-out chr22 genes ({time.time()-t1:.0f}s)")

    # ---- summary + comparison ---------------------------------------------------
    def med_ci(v, n=2000):
        rng = np.random.default_rng(0); v = np.asarray(v)
        m = [np.median(rng.choice(v, len(v), replace=True)) for _ in range(n)]
        return np.median(v), np.percentile(m, 2.5), np.percentile(m, 97.5)
    from scipy.stats import wilcoxon
    mt, lo, hi = med_ci(df.rho_transfer)
    mb, blo, bhi = med_ci(df.rho_burden)
    print(f"\nZERO-SHOT held-out-gene cross-individual ρ (n={len(df)} genes):")
    print(f"  transfer model : median={mt:+.3f} [{lo:+.3f},{hi:+.3f}]  frac(ρ>0)={float((df.rho_transfer>0).mean()):.2f}")
    print(f"  burden (Σdosage): median={mb:+.3f} [{blo:+.3f},{bhi:+.3f}]")
    try:
        p = wilcoxon(df.rho_transfer, alternative="greater").pvalue
        print(f"  Wilcoxon ρ_transfer>0: p={p:.2e}")
    except Exception:
        pass

    # reference points from earlier phases
    enf = pd.read_csv(RESULTS / "phase5_enformer_zeroshot.csv")
    lin = pd.read_csv(RESULTS / "phase5_linear_baselines.csv")
    print(f"  [ref] Enformer zero-shot median ρ = {enf.rho_enformer_zeroshot.median():+.3f} (15 genes)")
    print(f"  [ref] per-gene elastic-net (WITH labels) median ρ = {lin.rho_enet.median():+.3f} (78 genes)")

    # ---- figure -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    series = [("Enformer\nzero-shot", enf.rho_enformer_zeroshot, "#C44E52"),
              ("burden\nzero-shot", df.rho_burden, "#8C8C8C"),
              ("transfer model\nzero-shot (ours)", df.rho_transfer, "#55A868"),
              ("elastic-net\n(with labels)", lin.rho_enet, "#4C72B0")]
    bp = ax.boxplot([s[1] for s in series], tick_labels=[s[0] for s in series],
                    showmeans=True, patch_artist=True)
    for patch, s in zip(bp["boxes"], series):
        patch.set_facecolor(s[2]); patch.set_alpha(0.8)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("per-gene cross-individual Spearman ρ")
    ax.set_title("Zero-shot held-out-gene transfer (chr22)\ngene-agnostic variant-effect model vs Enformer zero-shot")
    fig.tight_layout(); fig.savefig(RESULTS / "phase6_transfer_zeroshot.png", dpi=130)
    print("\nwrote results/phase6_transfer_zeroshot.csv, phase6_transfer_zeroshot.png")
    print("Phase 6 transfer model complete.")


if __name__ == "__main__":
    main()
