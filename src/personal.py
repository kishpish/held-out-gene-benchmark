"""Batched personal-sequence construction (METHODOLOGY §3.6).

Scales the single-individual `variants.build_haplotypes` to many individuals efficiently: one
remote `bcftools` stream per gene window returns the genotype matrix for ALL requested samples,
from which any individual's two phased haplotype windows are built by applying alt alleles.

Design decision (documented in docs/PHASE2_personal_sequences.md): we DO NOT materialize the
~11M personal sequences (12,360 genes × 449 individuals × 2 haplotypes × 196,608 bp ≈ multi-TB).
Instead the genotype matrix per window is cheap to stream/cache, and haplotype strings are built
on the fly when feeding the model; the reference-window features are cached once (hard rule 9).

SNV-only, biallelic, phased (hard rule 8). Per-variant REF concordance is asserted.
"""
from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from genome import Reference, revcomp

MAMBA_BIN = "/home/shadeform/.local/bin/micromamba"
MAMBA_ROOT = "/ephemeral/micromamba"
MAMBA_ENV = "cisgram"


def _mamba_env() -> dict:
    return dict(os.environ, MAMBA_ROOT_PREFIX=MAMBA_ROOT, HTS_ALLOW_REMOTE="1")


def write_sample_file(samples: list[str], path: Path) -> Path:
    path.write_text("\n".join(samples) + "\n")
    return path


@dataclass
class WindowGenotypes:
    """Genotype matrix for one gene window over a fixed sample list."""
    chrom: str
    tss1: int
    length: int
    strand: str
    window_start0: int
    samples: list[str]
    positions: np.ndarray          # (n_var,) 1-based POS
    ref_alleles: np.ndarray        # (n_var,) str
    alt_alleles: np.ndarray        # (n_var,) str
    gt: np.ndarray                 # (n_var, n_samp, 2) int8: 0=ref,1=alt,-1=missing
    ref_window: str                # + strand reference window (uppercase)

    @property
    def n_variants(self) -> int:
        return len(self.positions)


def stream_window_genotypes(ref: Reference, vcf_url: str, chrom: str, tss1: int,
                            samples: list[str], sample_file: Path,
                            length: int = 196_608, strand: str = "+") -> WindowGenotypes:
    """One bcftools stream → genotype matrix for all `samples` in the TSS-centered window."""
    start0 = tss1 - 1 - length // 2
    if start0 < 0:
        raise ValueError(f"window underflows contig (tss={tss1}, len={length})")
    end0 = start0 + length
    ref_window = ref.get_seq(chrom, start0, end0)
    assert len(ref_window) == length

    region = f"{chrom}:{start0 + 1}-{end0}"
    cmd = [MAMBA_BIN, "run", "-n", MAMBA_ENV, "bcftools", "query",
           "-r", region, "-S", str(sample_file),
           "-i", 'TYPE="snp" && N_ALT=1',
           "-f", "%POS\t%REF\t%ALT[\t%GT]\n", vcf_url]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         env=_mamba_env()).stdout

    positions, refs, alts, gts = [], [], [], []
    rw = ref_window  # + strand
    for line in out.splitlines():
        f = line.split("\t")
        pos1, r, a = int(f[0]), f[1], f[2]
        off = (pos1 - 1) - start0
        if off < 0 or off >= length:
            continue
        assert rw[off] == r, f"REF concordance fail @ {chrom}:{pos1}: VCF={r} ref={rw[off]}"
        row = np.full((len(samples), 2), -1, dtype=np.int8)
        for j, gtstr in enumerate(f[3:]):
            sep = "|" if "|" in gtstr else "/"
            al = gtstr.split(sep)
            if len(al) == 2:
                row[j, 0] = -1 if al[0] in (".", "") else (1 if al[0] == "1" else 0)
                row[j, 1] = -1 if al[1] in (".", "") else (1 if al[1] == "1" else 0)
        positions.append(pos1); refs.append(r); alts.append(a); gts.append(row)

    gt = np.stack(gts) if gts else np.zeros((0, len(samples), 2), np.int8)
    return WindowGenotypes(chrom=chrom, tss1=tss1, length=length, strand=strand,
                           window_start0=start0, samples=list(samples),
                           positions=np.array(positions, dtype=np.int64),
                           ref_alleles=np.array(refs), alt_alleles=np.array(alts),
                           gt=gt, ref_window=ref_window)


def haplotypes(wg: WindowGenotypes, sample: str) -> tuple[str, str]:
    """Build (hap1, hap2) model-orientation strings for one sample from the cached matrix."""
    j = wg.samples.index(sample)
    h1 = list(wg.ref_window)
    h2 = list(wg.ref_window)
    for i in range(wg.n_variants):
        off = int(wg.positions[i] - 1 - wg.window_start0)
        a = wg.alt_alleles[i]
        if wg.gt[i, j, 0] == 1:
            h1[off] = a
        if wg.gt[i, j, 1] == 1:
            h2[off] = a
    s1, s2 = "".join(h1), "".join(h2)
    if wg.strand == "-":
        s1, s2 = revcomp(s1), revcomp(s2)
    return s1, s2


def stream_dosage_matrix(vcf_url: str, chrom: str, start1: int, end1: int,
                         samples: list[str], sample_file: Path):
    """Reference-FREE: stream biallelic-SNV alt-dosages for `samples` in [start1,end1] (1-based).

    Returns (positions [n_snp], dosage [n_snp, n_samp] float32 with NaN for missing, alt_freq [n_snp]).
    No reference needed (used by cis-h² which only needs genotypes) → enables genome-wide runs without
    the full reference FASTA. Dosage = count of alt alleles per individual (0,1,2).
    """
    region = f"{chrom}:{start1}-{end1}"
    cmd = [MAMBA_BIN, "run", "-n", MAMBA_ENV, "bcftools", "query",
           "-r", region, "-S", str(sample_file),
           "-i", 'TYPE="snp" && N_ALT=1',
           "-f", "%POS[\t%GT]\n", vcf_url]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True,
                         env=_mamba_env()).stdout
    positions, rows = [], []
    for line in out.splitlines():
        f = line.split("\t")
        pos1 = int(f[0])
        dos = np.full(len(samples), np.nan, dtype=np.float32)
        for j, gt in enumerate(f[1:]):
            sep = "|" if "|" in gt else "/"
            al = gt.split(sep)
            if len(al) == 2 and al[0] not in (".", "") and al[1] not in (".", ""):
                dos[j] = (al[0] == "1") + (al[1] == "1")
        positions.append(pos1); rows.append(dos)
    if not rows:
        return np.array([], np.int64), np.zeros((0, len(samples)), np.float32), np.array([])
    D = np.stack(rows)
    af = np.nanmean(D, axis=1) / 2.0
    return np.array(positions, np.int64), D, af


def variant_stats(wg: WindowGenotypes) -> dict:
    """Per-window and per-individual personal-variation statistics."""
    gt = wg.gt
    if wg.n_variants == 0:
        return dict(n_snv=0, n_segregating=0, mean_alt_per_indiv=0.0,
                    mean_het_per_indiv=0.0, mean_homalt_per_indiv=0.0)
    valid = gt >= 0
    alt = (gt == 1)
    carrier = alt.any(axis=2)                      # (n_var, n_samp) individual carries alt
    ac = alt.sum(axis=(1, 2))                       # alt-allele count per variant
    an = valid.sum(axis=(1, 2))
    seg = ((ac > 0) & (ac < an)).sum()              # segregating among these samples
    het = (gt[:, :, 0] != gt[:, :, 1]) & valid.all(axis=2)
    homalt = (gt[:, :, 0] == 1) & (gt[:, :, 1] == 1)
    return dict(
        n_snv=int(wg.n_variants),
        n_segregating=int(seg),
        mean_alt_per_indiv=float(carrier.sum(axis=0).mean()),
        mean_het_per_indiv=float(het.sum(axis=0).mean()),
        mean_homalt_per_indiv=float(homalt.sum(axis=0).mean()),
    )
