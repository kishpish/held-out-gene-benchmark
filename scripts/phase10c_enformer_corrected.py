#!/usr/bin/env python
"""Phase 10c — matched Enformer zero-shot on the SUB-POPULATION-CORRECTED target.

Closes the loop from Phase 10b: the linear baselines were shown robust to sub-population correction, but
Enformer zero-shot (Phase 5b) was scored on the light target. Here we re-evaluate Enformer on the same 15
top-h² chr22 genes and 55 held-out EUR individuals, correlating its diploid GM12878-CAGE prediction with BOTH
the light target (sanity, should ≈ the Phase-5b 0.047) and the sub-pop-corrected target — confirming the
linear≫Enformer gap holds under a fully matched, ancestry-corrected comparison.

Run: .venv/bin/python scripts/phase10c_enformer_corrected.py [N_GENES]
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression

from paths import REF_DIR, PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
from genome import Reference
from personal import write_sample_file
import backbone
import eval_enformer

N_GENES = int(sys.argv[1]) if len(sys.argv) > 1 else 15


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    h2 = pd.read_csv(RESULTS / "phase3_cis_h2_chr22.csv").set_index("gene_id")
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    isplit = json.loads((SPLITS / "individual_split.json").read_text())

    eur = set(im.index[im.Superpopulation == "EUR"])
    train = [s for s in isplit["train"] if s in eur]
    test = [s for s in isplit["test"] if s in eur]
    samples = train + test
    Ptr = pd.get_dummies(im.loc[train, "Population"].astype("category"))
    Pall = pd.get_dummies(im.loc[samples, "Population"].astype("category")).reindex(columns=Ptr.columns, fill_value=0)
    sf = write_sample_file(test, CACHE / "p10c_test.txt")
    url = kgp_vcf_url("chr22")
    genes = h2.loc[[g for g in selected if g in h2.index]].sort_values("h2_eur", ascending=False).head(N_GENES)
    print(f"Enformer on corrected target: {len(genes)} genes × {len(test)} held-out EUR individuals")
    backbone.load_enformer()

    rows = []; t0 = time.time()
    for gid, g in genes.iterrows():
        gm_g = gm.loc[gid]
        ev = eval_enformer.evaluate_gene(ref, url, "chr22", int(gm_g.tss), gm_g.strand, test, sf, batch_size=6)
        y = target.loc[gid, samples].to_numpy(float)
        # sub-pop correction fit on TRAIN, applied to all; take test slice
        lr = LinearRegression().fit(Ptr.to_numpy(float), y[:len(train)])
        yc = y - lr.predict(Pall.to_numpy(float))
        y_light = target.loc[gid, test].to_numpy(float)
        y_corr = yc[len(train):]
        cage = ev.cage_diploid
        rho_light = 0.0 if np.std(cage) < 1e-9 else float(spearmanr(cage, y_light).statistic)
        rho_corr = 0.0 if np.std(cage) < 1e-9 else float(spearmanr(cage, y_corr).statistic)
        rows.append(dict(gene=gm_g.gene_name, gene_id=gid, rho_light=rho_light, rho_corrected=rho_corr))
        print(f"  {gm_g.gene_name:12s} rho_light={rho_light:+.3f}  rho_corrected={rho_corr:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase10c_enformer_corrected.csv", index=False)
    print(f"\n({time.time()-t0:.0f}s) Enformer zero-shot median ρ: "
          f"light={df.rho_light.median():+.3f}  sub-pop-corrected={df.rho_corrected.median():+.3f}")
    print(f"  vs linear (sub-pop-corrected, Phase 10b): elastic-net +0.229, lead-eQTL +0.225")
    print("Phase 10c complete.")


if __name__ == "__main__":
    main()
