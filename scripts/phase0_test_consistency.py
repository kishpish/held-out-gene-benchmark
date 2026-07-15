#!/usr/bin/env python
"""Phase 0 — build-consistency + gene/individual selection driver.

Documented in docs/PHASE0_environment_inventory_smoketest.md §6. Verifies GRCh38 build
consistency by REF-vs-reference concordance on streamed 1000G SNVs, and deterministically
selects the smoke-test gene (highest-expressed window-safe chr22 gene present in GEUVADIS).

Run: .venv/bin/python scripts/phase0_test_consistency.py
Expected: reference contig ['22']; 447 chr22 protein-coding genes; 462 GEUVADIS individuals;
selected gene RPL3; concordance 581/581 SNVs, 0 mismatches.
"""
import sys
import gzip
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd

from paths import REF_DIR, GENCODE_DIR, GEUVADIS_DIR, kgp_vcf_url
from genome import Reference, assert_build_is_grch38, assert_vcf_ref_matches_reference
from annotation import load_genes


def main() -> None:
    assert_build_is_grch38()
    print("[OK] build == GRCh38")

    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.chromosome.22.fa")
    print("[OK] reference contigs:", ref.contigs(), "style:", ref.style)

    genes = load_genes(GENCODE_DIR / "gencode.v44.annotation.gtf.gz", "chr22")
    print(f"[OK] {len(genes)} protein-coding genes on chr22")

    # GEUVADIS: header -> unique 1000G ids; mean RPKM per chr22 gene (base id)
    gz = GEUVADIS_DIR / "GD660.GeneQuantRPKM.txt.gz"
    with gzip.open(gz, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    sample_cols = header[4:]
    kgp_ids = sorted({c.split(".")[0] for c in sample_cols})
    print(f"[OK] GEUVADIS: {len(sample_cols)} libraries, {len(kgp_ids)} unique 1000G individuals")

    rows = []
    with gzip.open(gz, "rt") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[2] != "22":                       # GEUVADIS Chr col (label only; coords are hg19)
                continue
            rows.append((f[0].split(".")[0], sum(float(x) for x in f[4:]) / len(sample_cols)))
    expr = pd.DataFrame(rows, columns=["gene_id_base", "mean_rpkm"]).groupby("gene_id_base").mean()
    print(f"[OK] GEUVADIS chr22 genes: {len(expr)}")

    # join by base id; require full-window flanking
    L = 196_608
    half = L // 2
    chrlen = len(ref._fa[ref.contigs()[0]])
    g = genes.merge(expr, on="gene_id_base", how="inner")
    g = g[(g.tss > half) & (g.tss < chrlen - half)].sort_values("mean_rpkm", ascending=False)
    print(f"[OK] candidate genes (in GEUVADIS, window-safe): {len(g)}")
    print(g[["gene_name", "gene_id", "strand", "tss", "mean_rpkm"]].head(8).to_string())

    top = g.iloc[0]
    print("\n=== SELECTED smoke gene ===")
    print(dict(gene_name=top.gene_name, gene_id=top.gene_id, strand=top.strand, tss=int(top.tss)))

    # build-consistency: REF concordance in a 20 kb window around the TSS (streamed 1000G)
    region = f"chr22:{int(top.tss) - 10000}-{int(top.tss) + 10000}"
    res = assert_vcf_ref_matches_reference(ref, kgp_vcf_url("chr22"), region, min_snvs=25)
    print("[OK] REF-vs-reference concordance:", res)


if __name__ == "__main__":
    main()
