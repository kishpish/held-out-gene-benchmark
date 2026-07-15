#!/usr/bin/env bash
# Phase 6 — fetch curated LCL/B-cell-relevant TF motifs (JASPAR PFMs) for the transferable
# variant-effect features. Documented in docs/PHASE6_transferable_variant_effect.md.
set -uo pipefail
R=/ephemeral/cis-grammar/raw/jaspar; mkdir -p "$R"
# name -> JASPAR matrix id (2024)
declare -A M=(
  [CTCF]=MA0139.1 [SPI1]=MA0080.5 [EBF1]=MA0154.4 [NFKB1]=MA0105.4 [IRF1]=MA0050.2
  [SP1]=MA0079.4 [YY1]=MA0095.2 [RUNX3]=MA0684.2 [POU2F2]=MA0507.1 [BATF]=MA1634.1 )
for name in "${!M[@]}"; do
  id=${M[$name]}
  code=$(timeout 30 curl -sG "https://jaspar.elixir.no/api/v1/matrix/$id/" \
           -H "Accept: application/json" -o "$R/$name.$id.json" -w "%{http_code}" 2>/dev/null)
  echo "  $name ($id): HTTP $code"
done
echo "motifs in $R: $(ls -1 "$R"/*.json | wc -l)"
