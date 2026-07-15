#!/usr/bin/env bash
# Phase 0 — fetch the smoke-test-minimal data (~123 MB) to /ephemeral.
# Documented in docs/PHASE0_environment_inventory_smoketest.md §5.8. 1000G is NOT downloaded
# (streamed at runtime). Enformer weights auto-download on first model load.
set -euo pipefail

R=/ephemeral/cis-grammar/raw
MM=/home/shadeform/.local/bin/micromamba
export MAMBA_ROOT_PREFIX=/ephemeral/micromamba
mkdir -p "$R"/{reference,gencode,geuvadis,enformer}

echo "[1/4] Ensembl GRCh38 chr22 FASTA (~11 MB)"
curl -s -o "$R/reference/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz" \
  "http://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz"
gzip -dkf "$R/reference/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz"
"$MM" run -n cisgram samtools faidx "$R/reference/Homo_sapiens.GRCh38.dna.chromosome.22.fa"

echo "[2/4] GENCODE v44 annotation GTF (48 MB)"
curl -s -o "$R/gencode/gencode.v44.annotation.gtf.gz" \
  "http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"

echo "[3/4] GEUVADIS GD660 expression matrix (59 MB)"
curl -s -o "$R/geuvadis/GD660.GeneQuantRPKM.txt.gz" \
  "https://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEUV/E-GEUV-1/analysis_results/GD660.GeneQuantRPKM.txt.gz"

echo "[4/4] Enformer human target-track table (<1 MB)"
curl -s -o "$R/enformer/targets_human.txt" \
  "https://raw.githubusercontent.com/calico/basenji/master/manuscripts/cross2020/targets_human.txt"

echo "[✓] Sizes:"; ls -lh "$R"/reference/*.fa "$R"/gencode/*.gz "$R"/geuvadis/*.gz "$R"/enformer/*.txt
