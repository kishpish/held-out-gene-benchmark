#!/usr/bin/env python
"""Phase 2 — validate & characterize batched personal-sequence construction.

Documented in docs/PHASE2_personal_sequences.md. Runs on chr22 (reference already local; genome-wide
scaling needs the full GRCh38 reference — a flagged gate). Does three things:
  1. CORRECTNESS: batched builder (`personal.haplotypes`) must exactly match the single-individual
     path (`variants.build_haplotypes`) for the same sample, and one-hot/revcomp shapes must be right.
  2. CHARACTERIZATION: per-gene personal cis-variation statistics over a sample of chr22 genes
     (SNVs/window, segregating variants, het & alt counts per individual) — how much signal exists.
  3. THROUGHPUT: timing → extrapolation to the full 12,360-gene working set.

Writes results/phase2_variant_stats.csv. Run: .venv/bin/python scripts/phase2_validate_sequences.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd

from paths import REF_DIR, PROCESSED, RESULTS, CACHE, kgp_vcf_url
from genome import Reference
from variants import build_haplotypes
from sequence import one_hot
import personal

N_GENES = 50
SEED = 0
LENGTH = 196_608


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.chromosome.22.fa")
    chrlen = len(ref._fa[ref.contigs()[0]])
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    samples = list(im.index)
    url = kgp_vcf_url("chr22")

    half = LENGTH // 2
    chr22 = gm[(gm.chrom == "chr22") & (gm.tss > half) & (gm.tss < chrlen - half)].copy()
    genes = chr22.sample(min(N_GENES, len(chr22)), random_state=SEED).sort_values("tss")
    print(f"chr22 window-safe genes: {len(chr22)}; validating on {len(genes)}; individuals: {len(samples)}")

    sample_file = personal.write_sample_file(samples, CACHE / "geuvadis_449_samples.txt")

    # ---- 1. CORRECTNESS: batched vs single-individual path ----------------------
    g0 = genes.iloc[0]
    wg = personal.stream_window_genotypes(ref, url, "chr22", int(g0.tss), samples, sample_file,
                                          length=LENGTH, strand=g0.strand)
    for s in samples[:3]:
        h1b, h2b = personal.haplotypes(wg, s)
        single = build_haplotypes(ref, url, "chr22", int(g0.tss), s, length=LENGTH, strand=g0.strand)
        assert h1b == single.hap1 and h2b == single.hap2, f"batched != single for {s}"
        oh = one_hot(h1b)
        assert oh.shape == (LENGTH, 4)
    print(f"[1] CORRECTNESS OK: batched == single-individual path for gene {g0.gene_name} "
          f"({g0.strand} strand); one-hot shape {oh.shape}")

    # ---- 2 & 3. CHARACTERIZATION + THROUGHPUT -----------------------------------
    rows = []
    t0 = time.time()
    for _, g in genes.iterrows():
        t = time.time()
        wg = personal.stream_window_genotypes(ref, url, "chr22", int(g.tss), samples, sample_file,
                                              length=LENGTH, strand=g.strand)
        st = personal.variant_stats(wg)
        st.update(gene=g.gene_name, gene_id=g.name, tss=int(g.tss), strand=g.strand,
                  sec=round(time.time() - t, 2))
        rows.append(st)
    df = pd.DataFrame(rows)
    elapsed = time.time() - t0

    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "phase2_variant_stats.csv", index=False)

    print("\n[2] Personal cis-variation per 196,608 bp window (chr22 sample):")
    for col in ["n_snv", "n_segregating", "mean_alt_per_indiv", "mean_het_per_indiv", "mean_homalt_per_indiv"]:
        print(f"    {col:22s} median={df[col].median():8.1f}  mean={df[col].mean():8.1f}  "
              f"min={df[col].min():.0f}  max={df[col].max():.0f}")

    per_gene = elapsed / len(genes)
    print(f"\n[3] THROUGHPUT: {len(genes)} genes in {elapsed:.0f}s = {per_gene:.2f}s/gene "
          f"(all {len(samples)} individuals per stream).")
    print(f"    Extrapolation to 12,360 genes (genotype streaming only, 1 CPU): "
          f"~{per_gene*12360/60:.0f} min; parallelizable across the 28 cores.")
    print(f"\n    wrote results/phase2_variant_stats.csv")
    print("Phase 2 validation complete.")


if __name__ == "__main__":
    main()
