#!/usr/bin/env python
"""Phase 0 — sensitivity control for the build-consistency check.

Documented in docs/PHASE0_environment_inventory_smoketest.md §6. Confirms the REF-vs-reference
concordance assertion is genuinely sensitive to coordinate errors: comparing the same streamed
1000G SNVs against the reference base at pos+1 (a deliberate off-by-one) must drop concordance
far below 100%. If a 1 bp shift still matched, the check would be meaningless.

Run: .venv/bin/python scripts/phase0_check_sensitivity.py
Expected: correct-offset 581/581 (100%); shifted-by-1 ~147/581 (~25%, ≈ random over 4 bases).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from paths import REF_DIR, kgp_vcf_url
from genome import Reference, stream_snvs


def main() -> None:
    ref = Reference(REF_DIR / "Homo_sapiens.GRCh38.dna.chromosome.22.fa")
    url = kgp_vcf_url("chr22")
    region = "chr22:39309623-39329623"          # RPL3 TSS ± 10 kb

    correct = shifted = n = 0
    for chrom, pos1, r, a in stream_snvs(url, region):
        n += 1
        if ref.base_at(chrom, pos1) == r:
            correct += 1
        if ref.base_at(chrom, pos1 + 1) == r:   # deliberate off-by-one
            shifted += 1

    print(f"n={n}  correct-offset={correct}/{n} ({100*correct/n:.1f}%)  "
          f"shifted-by-1={shifted}/{n} ({100*shifted/n:.1f}%)")
    ok = correct == n and shifted < n * 0.5
    print("SENSITIVITY OK: check catches coordinate errors" if ok
          else "WARNING: check not sensitive")
    assert ok, "build-consistency check failed its sensitivity control"


if __name__ == "__main__":
    main()
