"""Enformer zero-shot cross-individual evaluation (METHODOLOGY §5.1 baseline #5).

For a gene, build each individual's two haplotype windows, run Enformer, read the GM12878 CAGE track at
the TSS bin, combine haplotypes (diploid = mean). The per-individual diploid CAGE is then correlated across
individuals with the centered expression target → the zero-shot cross-individual ρ. This reproduces the
documented failure (Huang 2023 / Sasse 2023): pretrained models show near-zero / wrong-direction personal
signal even where a linear model (fit per gene) does well.

The reference-window pass is cached once per gene (hard rule 9). Forwards are batched (bf16, H100).
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from genome import Reference
from sequence import one_hot
import personal
import backbone


@dataclass
class GeneEnformerEval:
    gene_id: str
    samples: list
    cage_diploid: np.ndarray     # (n_samp,) diploid CAGE at TSS bin
    cage_ref: float              # reference-window CAGE at TSS bin
    n_variants: int


def _batched_cage(seqs: list[str], track: int, bin_idx: int, batch_size: int = 6,
                  device: str = "cuda") -> np.ndarray:
    """Run Enformer on a list of sequences, return CAGE-track value at the TSS bin for each."""
    out = np.empty(len(seqs), dtype=np.float32)
    for i in range(0, len(seqs), batch_size):
        chunk = seqs[i:i + batch_size]
        oh = np.stack([one_hot(s) for s in chunk]).astype(np.float32)   # (B, L, 4)
        pred = backbone.predict_human_batch(oh, device=device)          # (B, 896, 5313)
        out[i:i + len(chunk)] = pred[:, bin_idx, track]
    return out


def evaluate_gene(ref: Reference, vcf_url: str, chrom: str, tss1: int, strand: str,
                  samples: list[str], sample_file, track: int = backbone.LCL_CAGE,
                  length: int = 196_608, batch_size: int = 6, device: str = "cuda") -> GeneEnformerEval:
    wg = personal.stream_window_genotypes(ref, vcf_url, chrom, tss1, samples, sample_file,
                                          length=length, strand=strand)
    # reference pass (cached once)
    ref_seq = personal.revcomp(wg.ref_window) if strand == "-" else wg.ref_window
    cage_ref = float(_batched_cage([ref_seq], track, backbone.CENTER_BIN, 1, device)[0])
    # personal haplotypes: build both, run, diploid-average
    seqs, order = [], []
    for s in samples:
        h1, h2 = personal.haplotypes(wg, s)
        seqs.append(h1); seqs.append(h2); order.append(s)
    cage = _batched_cage(seqs, track, backbone.CENTER_BIN, batch_size, device)
    diploid = (cage[0::2] + cage[1::2]) / 2.0
    return GeneEnformerEval(gene_id="", samples=order, cage_diploid=diploid,
                            cage_ref=cage_ref, n_variants=wg.n_variants)
