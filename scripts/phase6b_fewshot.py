#!/usr/bin/env python
"""Phase 6b — few-shot cross-individual prediction on held-out genes (METHODOLOGY §6.4, Gate-B fallback).

Documented in docs/PHASE6b_fewshot.md. For each held-out chr22 learnable gene, hold a fixed disjoint QUERY
set (EUR test individuals) and draw k SUPPORT individuals from the EUR train pool. Characterize per-gene
cross-individual ρ vs k for: lead-eQTL, elastic-net (all SNPs), elastic-net on the top-M SNPs prioritized by
the Phase-6 transfer model's |β̂| (prior-guided), and a random-M control. Answers: how many labels does a
held-out gene need, and does the transferable prior reduce that number?

CPU only. Run: .venv/bin/python scripts/phase6b_fewshot.py
"""
import os
# Pin BLAS to 1 thread per worker BEFORE numpy import — otherwise 8 ThreadPool workers × multi-threaded
# BLAS oversubscribe the cores and the elastic-net sweep slows to a crawl.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
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
import joblib

from paths import REF_DIR, PROCESSED, RESULTS, SPLITS, CACHE, RAW, kgp_vcf_url
from genome import Reference
from personal import stream_window_genotypes, write_sample_file
import motifs as motif_mod
import transfer_model as tm
import fewshot

LENGTH = 196_608
MAF_MIN = 0.05
K_GRID = [5, 10, 20, 40, 80, 160]
M_TOP = 50
REPEATS = 4
WORKERS = 10
N_GENES = 40          # top-h² held-out genes (keeps the sweep tractable)
MAX_VAR = 250         # cap common variants per gene (highest-heterozygosity) for speed


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.primary_assembly.fa")
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    motif_list = motif_mod.load_motifs(RAW / "jaspar")
    model = joblib.load(PROCESSED / "phase6_transfer_model.joblib")["model"]
    isplit = json.loads((SPLITS / "individual_split.json").read_text())
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    # top-h² held-out genes (clearest ceiling; keeps the sweep tractable)
    h2 = pd.read_csv(RESULTS / "phase3_cis_h2_chr22.csv").set_index("gene_id")
    selected = list(h2.loc[[g for g in selected if g in h2.index]]
                    .sort_values("h2_eur", ascending=False).head(N_GENES).index)

    eur = set(im.index[im.Superpopulation == "EUR"])
    support = [s for s in isplit["train"] if s in eur]     # support POOL (249)
    query = [s for s in isplit["test"] if s in eur]        # fixed query (55)
    samples = support + query
    sup_idx = np.arange(len(support)); qry_idx = np.arange(len(support), len(samples))
    sf = write_sample_file(samples, CACHE / "p6b_all_eur.txt")
    print(f"held-out genes: {len(selected)} | support pool {len(support)} | query {len(query)} | "
          f"k grid {K_GRID}")

    def prep(gid):
        g = gm.loc[gid]
        try:
            wg = stream_window_genotypes(ref, kgp_vcf_url("chr22"), "chr22", int(g.tss), samples, sf,
                                         length=LENGTH, strand="+")
            if wg.n_variants == 0:
                return None
            gt = wg.gt.astype(np.float32); gt[gt < 0] = np.nan
            D = np.nansum(gt, axis=2)
            af = np.nanmean(D, axis=1) / 2.0
            keep = (af >= MAF_MIN) & (af <= 1 - MAF_MIN)
            idx = np.where(keep)[0]
            if len(idx) < 5:
                return None
            # cap to the MAX_VAR most-variable (highest-heterozygosity) common SNVs for speed
            if len(idx) > MAX_VAR:
                het = 2 * af[idx] * (1 - af[idx])
                idx = idx[np.argsort(het)[::-1][:MAX_VAR]]
            X = np.empty((len(samples), len(idx)), np.float32)
            beta_abs = np.empty(len(idx), np.float32)
            cols = 0
            for i in idx:
                f = tm.variant_features_win(wg.ref_window, wg.window_start0, int(wg.positions[i]),
                                            wg.ref_alleles[i], wg.alt_alleles[i], int(g.tss), g.strand,
                                            motif_list, float(af[i]))
                if f is None:
                    continue
                row = D[i].copy(); m = np.nanmean(row); row[np.isnan(row)] = m
                X[:, cols] = row
                beta_abs[cols] = abs(float(model.predict(f[None, :])[0]))
                cols += 1
            if cols < 5:
                return None
            X = X[:, :cols]; beta_abs = beta_abs[:cols]
            y = target.loc[gid, samples].to_numpy(float)
            res = {}
            for k in K_GRID:
                res[k] = fewshot.fewshot_gene(X, y, sup_idx, qry_idx, k, beta_abs=beta_abs,
                                              M=M_TOP, repeats=REPEATS, seed=0)
            return gid, res
        except Exception:
            return None

    t0 = time.time(); rows = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(prep, gid) for gid in selected]):
            r = f.result()
            if r is not None:
                gid, res = r
                for k, methods in res.items():
                    rows.append(dict(gene_id=gid, k=k, **methods))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase6b_fewshot.csv", index=False)
    print(f"evaluated {df.gene_id.nunique()} genes across k-grid ({time.time()-t0:.0f}s)")

    # ---- aggregate + figure -----------------------------------------------------
    methods = ["lead_eqtl", "enet_all", "enet_transfer_topM", "enet_random_topM"]
    agg = df.groupby("k")[methods].median()
    print("\nmedian per-gene cross-individual ρ vs k (held-out genes):")
    print(agg.round(3).to_string())

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    styles = {"lead_eqtl": ("#DD8452", "lead-eQTL"),
              "enet_all": ("#4C72B0", "elastic-net (all SNPs)"),
              "enet_transfer_topM": ("#55A868", f"elastic-net (transfer top-{M_TOP})"),
              "enet_random_topM": ("#8C8C8C", f"elastic-net (random {M_TOP})")}
    for m in methods:
        c, lab = styles[m]
        ls = "--" if m == "enet_random_topM" else "-"
        ax.plot(agg.index, agg[m], marker="o", ms=4, lw=1.8, ls=ls, color=c, label=lab)
    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xticks(K_GRID); ax.set_xticklabels(K_GRID)
    ax.set_xlabel("k labeled support individuals (per held-out gene)")
    ax.set_ylabel("median per-gene cross-individual ρ (query)")
    ax.set_title("Few-shot on held-out genes: ρ vs number of labels\n(does the transferable prior reduce labels needed?)")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(RESULTS / "phase6b_fewshot.png", dpi=130)
    print("\nwrote results/phase6b_fewshot.csv, phase6b_fewshot.png")
    print("Phase 6b few-shot complete.")


if __name__ == "__main__":
    main()
