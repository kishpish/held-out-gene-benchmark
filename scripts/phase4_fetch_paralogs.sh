#!/usr/bin/env bash
# Phase 4 — fetch the Ensembl paralog table (protein-coding, autosomes) for the homology audit.
# Documented in docs/PHASE4_splits_and_leakage_audit.md. One-time BioMart fetch (per-chromosome),
# cached to data/raw/meta/ensembl_paralogs_autosomes.tsv (base gene ids, both directions present).
set -uo pipefail
R=/ephemeral/cis-grammar/raw/meta; mkdir -p "$R"
OUT="$R/ensembl_paralogs_autosomes.tsv"; : > "$OUT"
for c in $(seq 1 22); do
  Q="<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE Query>\
<Query virtualSchemaName=\"default\" formatter=\"TSV\" header=\"0\" uniqueRows=\"1\" datasetConfigVersion=\"0.6\">\
<Dataset name=\"hsapiens_gene_ensembl\" interface=\"default\">\
<Filter name=\"chromosome_name\" value=\"$c\"/>\
<Filter name=\"biotype\" value=\"protein_coding\"/>\
<Attribute name=\"ensembl_gene_id\"/>\
<Attribute name=\"hsapiens_paralog_ensembl_gene\"/></Dataset></Query>"
  n=$(timeout 120 curl -s -G "https://www.ensembl.org/biomart/martservice" \
        --data-urlencode "query=$Q" 2>/dev/null | awk 'NF==2' | tee -a "$OUT" | wc -l)
  echo "chr$c: $n pairs"
done
echo "TOTAL pairs: $(wc -l < "$OUT")"
