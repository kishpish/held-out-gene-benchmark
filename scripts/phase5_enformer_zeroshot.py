#!/usr/bin/env python
"""Phase 5b — Enformer ZERO-SHOT cross-individual baseline (reproduce the failure; Gate A).

Documented in docs/PHASE5_baselines_and_failure_reproduction.md. For the top-h² learnable chr22 genes,
run pretrained Enformer on each held-out EUR individual's two haplotype windows, read GM12878 CAGE at the
TSS bin, diploid-average, and correlate across individuals with the centered target. Compares against the
Phase-5a linear baselines on the SAME genes/individuals → the pretrained model should sit near ρ≈0 while a
per-gene linear model does well: the documented held-out failure.

SCOPED to stay within the interactive compute budget (default 15 genes × EUR test individuals × 2 haplotypes).
The full-scale run (all selected genes, and the fine-tuned-Enformer comparison) is flagged for sign-off.

Run: .venv/bin/python scripts/phase5_enformer_zeroshot.py [N_GENES]
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
from scipy.stats import spearmanr

from paths import REF_DIR, PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
from genome import Reference
from personal import write_sample_file
import backbone
import eval_enformer

N_GENES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BATCH = 6


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    h2 = pd.read_csv(RESULTS / "phase3_cis_h2_chr22.csv").set_index("gene_id")
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    isplit = json.loads((SPLITS / "individual_split.json").read_text())

    eur = set(im.index[im.Superpopulation == "EUR"])
    test = [s for s in isplit["test"] if s in eur]              # held-out EUR individuals
    sf = write_sample_file(test, CACHE / "phase5b_eur_test.txt")
    url = kgp_vcf_url("chr22")

    # top-h² selected genes (clearest ceiling)
    genes = h2.loc[[g for g in selected if g in h2.index]].sort_values("h2_eur", ascending=False).head(N_GENES)
    print(f"Enformer zero-shot: {len(genes)} genes × {len(test)} held-out EUR individuals "
          f"× 2 haplotypes = {len(genes)*len(test)*2} forwards (batch {BATCH})")

    # warm up model (also downloads/loads weights)
    backbone.load_enformer()
    rows = []
    t0 = time.time()
    for gid, g in genes.iterrows():
        t = time.time()
        ev = eval_enformer.evaluate_gene(ref, url, "chr22", int(g.tss), g.strand, test, sf,
                                         batch_size=BATCH)
        y = target.loc[gid, test].to_numpy(float)
        rho = 0.0 if np.std(ev.cage_diploid) < 1e-9 else float(spearmanr(ev.cage_diploid, y).statistic)
        rows.append(dict(gene_id=gid, gene=g.gene_name, h2_eur=float(g.h2_eur),
                         n_variants=ev.n_variants, cage_ref=ev.cage_ref,
                         cage_std=float(np.std(ev.cage_diploid)), rho_enformer_zeroshot=rho))
        print(f"  {g.gene_name:12s} h²={g.h2_eur:.2f}  ρ_enformer={rho:+.3f}  "
              f"(CAGE sd={np.std(ev.cage_diploid):.2f}, {time.time()-t:.0f}s)")
    df = pd.DataFrame(rows).set_index("gene_id")
    df.to_csv(RESULTS / "phase5_enformer_zeroshot.csv")
    print(f"\ntotal {time.time()-t0:.0f}s | median ρ_enformer_zeroshot = {df.rho_enformer_zeroshot.median():+.3f}")

    # ---- comparison figure vs linear baselines (SAME genes) ---------------------
    lin = pd.read_csv(RESULTS / "phase5_linear_baselines.csv").set_index("gene_id")
    common = df.index.intersection(lin.index)
    comp = pd.DataFrame({
        "lead-eQTL": lin.loc[common, "rho_lead"],
        "elastic-net": lin.loc[common, "rho_enet"],
        "Enformer zero-shot": df.loc[common, "rho_enformer_zeroshot"],
    })
    print("\nmedian per-gene cross-individual ρ on the SAME", len(common), "genes:")
    print(comp.median().to_string())

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bp = ax.boxplot([comp[c] for c in comp.columns], tick_labels=list(comp.columns),
                    showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#DD8452", "#4C72B0", "#C44E52"]):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    for i, c in enumerate(comp.columns, 1):
        ax.scatter(np.full(len(comp), i) + np.random.default_rng(0).normal(0, 0.05, len(comp)),
                   comp[c], s=14, c="k", alpha=0.4, zorder=3)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("per-gene cross-individual Spearman ρ")
    ax.set_title(f"Held-out-gene failure reproduction (chr22, n={len(common)} learnable genes)\n"
                 "per-gene linear models predict; pretrained Enformer does not")
    fig.tight_layout(); fig.savefig(RESULTS / "phase5_failure_reproduction.png", dpi=130)
    print("\nwrote results/phase5_enformer_zeroshot.csv, phase5_failure_reproduction.png")
    print("Phase 5b Enformer zero-shot complete.")


if __name__ == "__main__":
    main()
