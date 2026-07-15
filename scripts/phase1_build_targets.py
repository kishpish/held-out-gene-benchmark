#!/usr/bin/env python
"""Phase 1 — build centered cross-individual expression targets from GEUVADIS.

Documented in docs/PHASE1_expression_targets.md. Produces (in data/processed/):
  - geuvadis_target_int.parquet   genes × individuals, centered target (light scheme)
  - geuvadis_log_rpkm.parquet     genes × individuals, log2(RPKM+1) (for PCA/covariates)
  - gene_meta.parquet             kept genes + GRCh38 TSS/strand/chrom from GENCODE
  - individual_meta.parquet       GEUVADIS individuals + population/superpopulation/pedigree
and (in results/): expression_pca_ancestry.png (diagnostic: expression PCs separate ancestries).

Run: .venv/bin/python scripts/phase1_build_targets.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import GEUVADIS_DIR, GENCODE_DIR, PROCESSED, RESULTS
from expression import build_targets
from annotation import load_genes
from metadata import ancestry_map
from covariates import ExpressionPCA


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    # 1. targets ------------------------------------------------------------------
    out = build_targets(GEUVADIS_DIR / "GD660.GeneQuantRPKM.txt.gz",
                        min_rpkm=0.1, min_frac=0.5)
    target = out["target_int"]; log_rpkm = out["log_rpkm"]
    print(f"[1] genes kept (QC): {target.shape[0]:,} | individuals: {target.shape[1]}")
    # sanity: per-gene INT target is centered (mean~0, sd~1)
    m = float(target.mean(axis=1).abs().max()); s = float(target.std(axis=1).mean())
    print(f"    target per-gene |mean|max={m:.3g}  mean sd={s:.3f}  (expect ~0 and ~1)")

    # 2. individual metadata ------------------------------------------------------
    amap = ancestry_map()
    indiv = list(target.columns)
    in_panel = [s for s in indiv if s in amap.index]
    print(f"[2] individuals matched to 1000G panel: {len(in_panel)}/{len(indiv)}")
    imeta = amap.loc[in_panel].copy()
    print("    superpopulation breakdown:")
    print(imeta.Superpopulation.value_counts().to_string().replace("\n", "\n      "))
    print("    population breakdown:")
    print(imeta.Population.value_counts().to_string().replace("\n", "\n      "))
    n_rel = int((~imeta.is_founder).sum())
    print(f"    non-founder (related) individuals present: {n_rel}")

    # restrict matrices to matched individuals (keep order)
    target = target[in_panel]; log_rpkm = log_rpkm[in_panel]

    # 3. GRCh38 TSS from GENCODE (autosomes) -------------------------------------
    genes = load_genes(GENCODE_DIR / "gencode.v44.annotation.gtf.gz", chrom=None)
    kept_base = pd.DataFrame({"gene_id": target.index.astype(str)})
    kept_base["gene_id_base"] = kept_base["gene_id"].str.split(".").str[0]
    kept_base["symbol"] = out["gene_info"]["symbol"].reindex(target.index).values
    gm = kept_base.merge(genes[["gene_id_base", "chrom", "strand", "tss", "gene_name"]],
                         on="gene_id_base", how="inner")
    gm = gm.drop_duplicates("gene_id").set_index("gene_id")
    # keep only genes with a GRCh38 autosomal TSS
    target = target.loc[target.index.intersection(gm.index)]
    log_rpkm = log_rpkm.loc[target.index]
    gm = gm.loc[target.index]
    print(f"[3] genes with GRCh38 autosomal TSS: {target.shape[0]:,}")
    print("    per-chromosome gene counts:",
          gm.chrom.value_counts().reindex([f'chr{i}' for i in range(1, 23)]).to_dict())

    # 4. expression-PCA diagnostic ------------------------------------------------
    # Fit on ALL matched individuals here ONLY for the diagnostic figure (not a target).
    pca = ExpressionPCA.fit(log_rpkm, in_panel, k=20)
    scores = pca.transform(log_rpkm, in_panel)
    ev = pca.pca.explained_variance_ratio_
    # Which PC best tracks ancestry? point-biserial corr of each PC with AFR indicator.
    afr = (imeta.loc[in_panel, "Superpopulation"] == "AFR").astype(float).values
    pc_corr = {c: abs(float(np.corrcoef(scores[c].values, afr)[0, 1])) for c in scores.columns}
    anc_pc = max(pc_corr, key=pc_corr.get)
    anc_i = int(anc_pc.replace("exprPC", "")) - 1
    print(f"[4] |corr(PC, AFR)| top: " +
          ", ".join(f"{k}={v:.2f}" for k, v in sorted(pc_corr.items(), key=lambda x: -x[1])[:5]))
    print(f"    most ancestry-associated PC = {anc_pc} (|r|={pc_corr[anc_pc]:.2f}, var={ev[anc_i]*100:.1f}%)")

    colors = {"FIN": "#4C72B0", "CEU": "#6FA0D8", "GBR": "#274C77", "TSI": "#8172B3", "YRI": "#DD8452"}
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for pop, sub in imeta.groupby("Population"):
        ids = sub.index
        ax.scatter(scores.loc[ids, "exprPC1"], scores.loc[ids, anc_pc],
                   s=18, alpha=0.85, label=f"{pop} (n={len(ids)})", c=colors.get(pop, "gray"))
    ax.set_xlabel(f"Expression PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"{anc_pc} ({ev[anc_i]*100:.1f}%) — most ancestry-associated")
    ax.set_title(f"GEUVADIS expression PCs: ancestry is a minor axis\n"
                 f"({anc_pc} vs AFR |r|={pc_corr[anc_pc]:.2f}) — genotype PCs needed", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULTS / "expression_pca_ancestry.png", dpi=130)
    print(f"    saved results/expression_pca_ancestry.png | PC1-10 var%:",
          [round(float(x)*100, 1) for x in ev[:10]])

    # 5. persist ------------------------------------------------------------------
    target.to_parquet(PROCESSED / "geuvadis_target_int.parquet")
    log_rpkm.to_parquet(PROCESSED / "geuvadis_log_rpkm.parquet")
    gm.to_parquet(PROCESSED / "gene_meta.parquet")
    imeta.to_parquet(PROCESSED / "individual_meta.parquet")
    print("[5] wrote data/processed/: geuvadis_target_int, geuvadis_log_rpkm, gene_meta, individual_meta (.parquet)")
    print("\nPhase 1 target build complete.")


if __name__ == "__main__":
    main()
