"""Per-individual phased haplotype construction over a model window (SNV-only, GRCh38).

Streams one sample's biallelic SNV genotypes from a remote 1000G VCF via bcftools, then
edits the reference window into two phased haplotype strings. Reference-allele concordance
is asserted per variant (hard rule 1). Indels/SVs excluded (hard rule 8).
"""
from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass, field

from genome import Reference, revcomp

MAMBA_BIN = "/home/shadeform/.local/bin/micromamba"
MAMBA_ROOT = "/ephemeral/micromamba"
MAMBA_ENV = "cisgram"


def _run(cmd: list[str]) -> str:
    env = dict(os.environ, MAMBA_ROOT_PREFIX=MAMBA_ROOT, HTS_ALLOW_REMOTE="1")
    return subprocess.run([MAMBA_BIN, "run", "-n", MAMBA_ENV, *cmd],
                          check=True, capture_output=True, text=True, env=env).stdout


def sample_snvs(vcf_url: str, chrom: str, start1: int, end1: int, sample: str):
    """Yield (pos1, ref, alt, gt_str) biallelic SNVs for `sample` in [start1, end1] (1-based)."""
    region = f"{chrom}:{start1}-{end1}"
    out = _run(["bcftools", "query", "-r", region, "-s", sample,
                "-i", 'TYPE="snp" && N_ALT=1',
                "-f", "%POS\t%REF\t%ALT[\t%GT]\n", vcf_url])
    for line in out.splitlines():
        pos, ref, alt, gt = line.split("\t")
        yield int(pos), ref, alt, gt


@dataclass
class HaplotypePair:
    chrom: str
    sample: str
    window_start0: int          # 0-based window start on + strand
    length: int
    strand: str
    hap1: str                   # model-orientation (revcomp'd if minus strand)
    hap2: str
    reference: str              # model-orientation reference window
    n_variants: int             # SNVs applied (in-window, sample carries alt on >=1 hap)
    n_het: int
    applied: list = field(default_factory=list)   # (pos1, ref, alt, gt)


def build_haplotypes(ref: Reference, vcf_url: str, chrom: str, tss1: int, sample: str,
                     length: int = 196_608, strand: str = "+") -> HaplotypePair:
    """Two phased haplotype sequences centered on the 1-based TSS.

    Variants applied on the + strand window; the whole window is reverse-complemented for
    minus-strand genes so the model always sees TSS-relative orientation.
    """
    start0 = tss1 - 1 - length // 2
    if start0 < 0:
        raise ValueError(f"window underflows contig start (tss={tss1}, len={length})")
    end0 = start0 + length
    ref_win = list(ref.get_seq(chrom, start0, end0))   # + strand, uppercase
    assert len(ref_win) == length, f"reference window length {len(ref_win)} != {length}"

    hap1 = ref_win.copy()
    hap2 = ref_win.copy()
    applied = []
    n_het = 0
    for pos1, r, a, gt in sample_snvs(vcf_url, chrom, start0 + 1, end0, sample):
        off = (pos1 - 1) - start0
        if off < 0 or off >= length:
            continue
        assert ref_win[off] == r, (
            f"REF concordance fail @ {chrom}:{pos1}: VCF REF={r} vs reference {ref_win[off]}")
        # phased genotype 'a|b' (accept unphased '/' but flag)
        sep = "|" if "|" in gt else "/"
        alleles = gt.split(sep)
        if len(alleles) != 2:
            continue
        left, right = alleles
        carries = False
        if left == "1":
            hap1[off] = a; carries = True
        if right == "1":
            hap2[off] = a; carries = True
        if carries:
            applied.append((pos1, r, a, gt))
            if left != right:
                n_het += 1

    h1 = "".join(hap1)
    h2 = "".join(hap2)
    rw = "".join(ref_win)
    if strand == "-":
        h1, h2, rw = revcomp(h1), revcomp(h2), revcomp(rw)

    return HaplotypePair(chrom=chrom, sample=sample, window_start0=start0, length=length,
                         strand=strand, hap1=h1, hap2=h2, reference=rw,
                         n_variants=len(applied), n_het=n_het, applied=applied)
