"""Reference-genome access and GRCh38 build-consistency assertions.

Hard rule 1: one build (GRCh38) everywhere. The strongest, self-validating check is
*concordance* — the REF allele of every 1000G variant must equal the reference-FASTA base
at that GRCh38 coordinate. If coordinates, build, or chr-naming were mismatched, REF would
not match. We also confirm GENCODE genes resolve on the expected chromosome/strand.

Coordinate convention here: **0-based, half-open** [start, end) for `get_seq`, matching BED /
kipoiseq. VCF POS is 1-based; convert with pos0 = POS - 1.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess

from pyfaidx import Fasta

from paths import GENOME_BUILD

# ---------------------------------------------------------------------------
COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq: str) -> str:
    return seq.translate(COMPLEMENT)[::-1]


def norm_chrom(chrom: str, style: str) -> str:
    """Normalize chromosome name. style='ensembl' -> '22'; style='ucsc' -> 'chr22'."""
    c = chrom[3:] if chrom.lower().startswith("chr") else chrom
    if style == "ensembl":
        return c
    if style == "ucsc":
        return f"chr{c}"
    raise ValueError(f"unknown style {style!r}")


@dataclass
class Reference:
    """pyfaidx-backed reference. Auto-detects Ensembl ('22') vs UCSC ('chr22') naming."""
    fasta_path: Path

    def __post_init__(self):
        self._fa = Fasta(str(self.fasta_path), sequence_always_upper=True, rebuild=False)
        keys = set(self._fa.keys())
        # detect naming style from what's actually in the file
        self._style = "ucsc" if any(k.startswith("chr") for k in keys) else "ensembl"

    @property
    def style(self) -> str:
        return self._style

    def contigs(self) -> list[str]:
        return list(self._fa.keys())

    def get_seq(self, chrom: str, start: int, end: int) -> str:
        """0-based half-open [start, end) on the + strand, uppercased."""
        key = norm_chrom(chrom, self._style)
        if key not in self._fa:
            raise KeyError(f"contig {key!r} (from {chrom!r}) not in reference {self.fasta_path.name}")
        return str(self._fa[key][start:end])

    def base_at(self, chrom: str, pos1: int) -> str:
        """Single base at 1-based VCF position."""
        return self.get_seq(chrom, pos1 - 1, pos1)


# --------------------------- build-consistency asserts ---------------------------
def assert_build_is_grch38() -> None:
    assert GENOME_BUILD == "GRCh38", f"project build must be GRCh38, got {GENOME_BUILD!r}"


def stream_snvs(vcf_url: str, region: str, mamba_env: str = "cisgram",
                mamba_bin: str = "/home/shadeform/.local/bin/micromamba",
                mamba_root: str = "/ephemeral/micromamba", limit: int | None = None):
    """Yield (chrom, pos1, ref, alt) biallelic SNVs from a remote (or local) VCF via tabix.

    Uses the bioconda `cisgram` env's tabix so htslib remote streaming is available.
    """
    import os
    env = dict(os.environ, MAMBA_ROOT_PREFIX=mamba_root, HTS_ALLOW_REMOTE="1")
    cmd = [mamba_bin, "run", "-n", mamba_env, "tabix", vcf_url, region]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    n = 0
    try:
        for line in proc.stdout:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, _id, ref, alt = f[0], int(f[1]), f[2], f[3], f[4]
            # biallelic SNV only
            if len(ref) == 1 and len(alt) == 1 and ref in "ACGT" and alt in "ACGT":
                yield chrom, pos, ref, alt
                n += 1
                if limit and n >= limit:
                    break
    finally:
        proc.terminate()


def assert_vcf_ref_matches_reference(ref: Reference, vcf_url: str, region: str,
                                     min_snvs: int = 25) -> dict:
    """Concordance check: every streamed SNV's REF must equal the reference base.

    This simultaneously validates: same build (GRCh38), correct coordinate convention
    (1-based VCF vs 0-based FASTA), and correct chr-name handling. Raises on any mismatch.
    """
    checked = mism = 0
    examples = []
    for chrom, pos1, refallele, alt in stream_snvs(vcf_url, region):
        rb = ref.base_at(chrom, pos1)
        checked += 1
        if rb != refallele:
            mism += 1
            if len(examples) < 5:
                examples.append((chrom, pos1, refallele, rb))
    assert checked >= min_snvs, (
        f"only {checked} SNVs streamed from {region}; expected >= {min_snvs} "
        f"(streaming or region problem)")
    assert mism == 0, (
        f"BUILD MISMATCH: {mism}/{checked} VCF REF alleles disagree with reference FASTA "
        f"in {region}. Examples (chrom,pos,vcf_ref,fasta_base): {examples}")
    return {"region": region, "snvs_checked": checked, "mismatches": mism}


if __name__ == "__main__":
    assert_build_is_grch38()
    print("GRCh38 build assert OK")
