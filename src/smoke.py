"""Vertical-slice smoke test: one gene, one individual, full path end-to-end.

Path: pick chr22 gene (in GEUVADIS, window-safe) -> stream individual's phased SNVs ->
build two haplotype windows + reference window (SNV-only, minus-strand revcomp) -> one-hot ->
Enformer forward (ref + both haplotypes, bf16 on H100) -> read GM12878 LCL tracks at TSS bin ->
report per-haplotype values, diploid mean, and reference-vs-personal difference.

Proves the pipeline before scaling. Not a training step.
"""
from __future__ import annotations
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from paths import (REF_DIR, GENCODE_DIR, GEUVADIS_DIR, RESULTS, kgp_vcf_url)
from genome import Reference
from annotation import load_genes
from variants import build_haplotypes
from sequence import one_hot
import backbone


def geuvadis_chr22(gz: Path):
    """(sample_libraries, unique_kgp_ids, mean_rpkm_by_gene_base) for chr22."""
    with gzip.open(gz, "rt") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    sample_cols = header[4:]
    kgp_ids = sorted({c.split(".")[0] for c in sample_cols})
    rows = []
    with gzip.open(gz, "rt") as fh:
        fh.readline()
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if f[2] != "22":
                continue
            rows.append((f[0].split(".")[0], sum(float(x) for x in f[4:]) / len(sample_cols)))
    expr = (pd.DataFrame(rows, columns=["gene_id_base", "mean_rpkm"])
            .groupby("gene_id_base").mean())
    return sample_cols, kgp_ids, expr


def select_gene(ref: Reference, length: int = 196_608):
    genes = load_genes(GENCODE_DIR / "gencode.v44.annotation.gtf.gz", "chr22")
    _, kgp_ids, expr = geuvadis_chr22(GEUVADIS_DIR / "GD660.GeneQuantRPKM.txt.gz")
    chrlen = len(ref._fa[ref.contigs()[0]]); half = length // 2
    g = genes.merge(expr, on="gene_id_base", how="inner")
    g = g[(g.tss > half) & (g.tss < chrlen - half)].sort_values("mean_rpkm", ascending=False)
    return g.iloc[0], kgp_ids


def run(sample: str = "HG00096", device: str = "cuda") -> dict:
    t0 = time.time()
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.chromosome.22.fa")
    gene, kgp_ids = select_gene(ref)
    assert sample in kgp_ids, f"{sample} not in GEUVADIS individuals"
    tss = int(gene.tss); strand = gene.strand
    url = kgp_vcf_url("chr22")

    hp = build_haplotypes(ref, url, "chr22", tss, sample, length=196_608, strand=strand)
    assert len(hp.reference) == len(hp.hap1) == len(hp.hap2) == 196_608
    # sanity: haplotypes differ from reference only where variants were applied
    d1 = sum(a != b for a, b in zip(hp.reference, hp.hap1))
    d2 = sum(a != b for a, b in zip(hp.reference, hp.hap2))

    preds = {}
    for name, seq in [("reference", hp.reference), ("hap1", hp.hap1), ("hap2", hp.hap2)]:
        oh = one_hot(seq)
        assert oh.shape == (196_608, 4)
        p = backbone.predict_human(oh, device=device)
        assert p.shape == (896, 5313) and np.isfinite(p).all()
        preds[name] = backbone.read_tss(p).tracks

    diploid = {k: (preds["hap1"][k] + preds["hap2"][k]) / 2 for k in preds["reference"]}
    diff = {k: diploid[k] - preds["reference"][k] for k in diploid}

    result = dict(
        gene=dict(name=gene.gene_name, gene_id=gene.gene_id, strand=strand, tss=tss),
        sample=sample,
        window=dict(length=196_608, start0=hp.window_start0, chrom="chr22"),
        variants=dict(applied=hp.n_variants, het=hp.n_het,
                      hap1_edits=d1, hap2_edits=d2),
        readout=dict(reference=preds["reference"], hap1=preds["hap1"], hap2=preds["hap2"],
                     diploid_mean=diploid, personal_minus_reference=diff),
        elapsed_sec=round(time.time() - t0, 1),
    )
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "smoke_test.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import pprint
    pprint.pprint(run())
