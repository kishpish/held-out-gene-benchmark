"""Central path configuration.

Persistent code/results live under PROJECT; bulk downloads and backbone feature
caches live on the /ephemeral scratch disk (symlinked into data/). Import these
constants everywhere instead of hard-coding paths.
"""
from __future__ import annotations
import os
from pathlib import Path

PROJECT = Path(os.environ.get("CISGRAM_PROJECT", "/home/shadeform/project"))

# Persistent (survives VM stop)
SRC = PROJECT / "src"
NOTEBOOKS = PROJECT / "notebooks"
CONFIGS = PROJECT / "configs"
SPLITS = PROJECT / "splits"
RESULTS = PROJECT / "results"
PROCESSED = PROJECT / "data" / "processed"

# Ephemeral scratch (re-buildable; may be wiped on VM stop)
EPHEMERAL = Path(os.environ.get("CISGRAM_EPHEMERAL", "/ephemeral/cis-grammar"))
RAW = EPHEMERAL / "raw"          # == PROJECT/data/raw (symlink)
CACHE = EPHEMERAL / "cache"      # == PROJECT/data/cache (symlink)

# Reference / annotation assets (downloaded on approval)
REF_DIR = RAW / "reference"
GENCODE_DIR = RAW / "gencode"
GEUVADIS_DIR = RAW / "geuvadis"
MAGE_DIR = RAW / "mage"

# Genome build is a hard project invariant.
GENOME_BUILD = "GRCh38"

# 1000G high-coverage (30x) phased VCFs, GRCh38, NYGC release — EBI FTP.
# Per-chromosome, tabix-indexed → remote streaming (no bulk download).
KGP_FTP_BASE = (
    "http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
    "1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
)
def kgp_vcf_url(chrom: str) -> str:
    """Remote VCF URL for a chromosome (e.g. 'chr22'). tabix streams from here."""
    c = chrom if chrom.startswith("chr") else f"chr{chrom}"
    return f"{KGP_FTP_BASE}/1kGP_high_coverage_Illumina.{c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"


def ensure_dirs() -> None:
    for d in (RESULTS, PROCESSED, SPLITS, CONFIGS, RAW, CACHE,
              REF_DIR, GENCODE_DIR, GEUVADIS_DIR, MAGE_DIR):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    print(f"PROJECT   = {PROJECT}")
    print(f"EPHEMERAL = {EPHEMERAL}")
    print(f"build     = {GENOME_BUILD}")
    print(f"kgp chr22 = {kgp_vcf_url('chr22')}")
