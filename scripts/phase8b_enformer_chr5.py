#!/usr/bin/env python
"""Phase 8b — reproduce the Gate-A gap on a SECOND chromosome (chr5). GPU.

Documented in docs/PHASE8_robustness.md. Takes the chr5 learnable genes selected in Phase 8a, runs Enformer
zero-shot cross-individual prediction on held-out EUR individuals, and compares to the linear baselines
(already computed in Phase 8a) on the same genes — confirming linear ≫ Enformer-zero-shot replicates off chr22.

Run: .venv/bin/python scripts/phase8b_enformer_chr5.py [N_GENES]
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
CHROM = "chr5"


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    p8 = pd.read_csv(RESULTS / "phase8a_multichrom.csv").set_index("gene_id")
    isplit = json.loads((SPLITS / "individual_split.json").read_text())

    eur = set(im.index[im.Superpopulation == "EUR"])
    test = [s for s in isplit["test"] if s in eur]
    sf = write_sample_file(test, CACHE / "p8b_test.txt")
    url = kgp_vcf_url(CHROM)

    learn = p8[(p8.chrom == CHROM) & (p8.h2 > 0.05) & (p8.p_h2 < 0.05)].sort_values("h2", ascending=False)
    genes = learn.head(N_GENES)
    print(f"{CHROM}: {len(learn)} learnable genes; Enformer zero-shot on top {len(genes)} × {len(test)} EUR test")

    backbone.load_enformer()
    rows = []; t0 = time.time()
    for gid, g in genes.iterrows():
        gm_g = gm.loc[gid]
        ev = eval_enformer.evaluate_gene(ref, url, CHROM, int(gm_g.tss), gm_g.strand, test, sf, batch_size=6)
        y = target.loc[gid, test].to_numpy(float)
        rho = 0.0 if np.std(ev.cage_diploid) < 1e-9 else float(spearmanr(ev.cage_diploid, y).statistic)
        rows.append(dict(gene_id=gid, gene=gm_g.gene_name, h2=float(g.h2),
                         rho_enformer=rho, rho_lead=float(g.rho_lead), rho_enet=float(g.rho_enet)))
        print(f"  {gm_g.gene_name:12s} h²={g.h2:.2f} enf={rho:+.3f} lead={g.rho_lead:+.3f} enet={g.rho_enet:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase8b_chr5_gateA.csv", index=False)
    print(f"\n{CHROM} medians ({time.time()-t0:.0f}s): lead={df.rho_lead.median():+.3f}  "
          f"enet={df.rho_enet.median():+.3f}  Enformer-zeroshot={df.rho_enformer.median():+.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    bp = ax.boxplot([df.rho_lead, df.rho_enet, df.rho_enformer],
                    tick_labels=["lead-eQTL", "elastic-net", "Enformer\nzero-shot"],
                    showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#DD8452", "#4C72B0", "#C44E52"]):
        patch.set_facecolor(c); patch.set_alpha(0.85)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("per-gene cross-individual ρ")
    ax.set_title(f"Gate A replicates on chr5 (n={len(df)} learnable genes)\nlinear predicts; Enformer zero-shot ≈ 0")
    fig.tight_layout(); fig.savefig(RESULTS / "phase8b_chr5_gateA.png", dpi=130)
    print("wrote results/phase8b_chr5_gateA.csv, phase8b_chr5_gateA.png")
    print("Phase 8b complete.")


if __name__ == "__main__":
    main()
