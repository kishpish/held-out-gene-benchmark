#!/usr/bin/env bash
# Phase 0 — environment probe.
# Read-only inspection of the host used to establish hardware/software facts recorded in
# docs/PHASE0_environment_inventory_smoketest.md §1. Safe to re-run.
set -uo pipefail

echo "=== HOST / DISK / MEM / CPU ==="
uname -a
df -h / /ephemeral 2>/dev/null
free -h
echo "CPU cores: $(nproc)"

echo "=== GPU ==="
nvidia-smi || echo "no nvidia-smi"

echo "=== Package managers ==="
which uv && uv --version
which micromamba && micromamba --version || echo "micromamba not yet installed"

echo "=== Genomics CLI presence (expect NONE before setup) ==="
for t in bcftools samtools tabix bgzip bedtools plink2 gcta64 liftOver; do
  printf "%-10s " "$t"; which "$t" 2>/dev/null || echo "NOT FOUND"
done

echo "=== Outbound connectivity ==="
for host in \
  http://ftp.ensembl.org \
  http://ftp.1000genomes.ebi.ac.uk \
  https://huggingface.co \
  https://zenodo.org ; do
  code=$(timeout 20 curl -sI "$host" 2>/dev/null | head -1)
  printf "%-45s %s\n" "$host" "${code:-UNREACHABLE}"
done
