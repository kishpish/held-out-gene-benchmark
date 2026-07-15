# Data Inventory Report — Phase 0 (notebook 00)

Generated 2026-07-05. All sources verified reachable via HEAD/streaming; **no bulk downloads performed
during inventory.** Project genome build = **GRCh38** (hard rule 1).

## Summary table

| Dataset | Role | Build | N | Size | Access | License |
|---|---|---|---|---|---|---|
| **1000G high-cov 30× phased** | personal haplotypes | GRCh38 | 3,202 (2,504+698 rel.) | 445 MB/chr (streamed) | tabix remote ✓ | open (IGSR/NYGC) |
| **GEUVADIS GD660** | expression (primary) | genes GRCh38-joinable | 660 cols / 462 indiv | 61 MB | HTTP ✓ | open research |
| **MAGE v1.0** | expression (ancestry axis) | GRCh38 | 731 / 26 pops | **59.5 GB zip** | Zenodo/Dropbox | CC-BY-4.0 |
| **GENCODE v44** | gene/TSS annotation | GRCh38 | — | 50 MB | HTTP ✓ | open |
| **Ensembl GRCh38 ref** | reference FASTA | GRCh38 | — | 882 MB full / ~12 MB per-chr | HTTP ✓ | open |
| **GTEx v8** | ⚠ external eQTL dir. only | GRCh38 | — | — | expr open / **genotypes dbGaP** | controlled |

---

## 1000 Genomes high-coverage (30×) phased VCFs — GRCh38, NYGC
- **URL base:** `http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/`
- Per-chromosome: `1kGP_high_coverage_Illumina.{chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz` (+`.tbi`).
- **chr22 = 445.7 MB**, `.tbi` present. **Header confirms 3,202 samples.**
- **Remote tabix VALIDATED:** streamed `chr22:23,180,000-23,185,000` in ~2.2 s → 160 variants, no bulk pull.
  This is the personal-haplotype source; we never download whole VCFs.
- chr naming: **`chr22`** (with prefix). Contains SNV+INDEL+SV; we filter **SNV-only** (rule 8).
- 698 related samples included → need kinship/pedigree for splits (rule 5). Pedigree file in same collection.
- Target path (streamed subsets only): `data/raw/kgp_windows/`.

## GEUVADIS (E-GEUV-1) — LCL RNA-seq
- **URL:** `https://ftp.ebi.ac.uk/pub/databases/microarray/data/experiment/GEUV/E-GEUV-1/analysis_results/GD660.GeneQuantRPKM.txt.gz` (**61 MB**, HTTP 200).
- ArrayExpress→BioStudies migration; the `analysis/` path 404s, **`analysis_results/` works**. GD462 variant 404s; **GD660 is the available matrix.**
- Format: TSV, cols = `TargetID` (Ensembl gene ID), `Gene_Symbol`, `Chr`, `Coord`, then **660 sample columns**
  named `HG00096.1.M_111124_6` (1000G individual ID = prefix before first `.`; 462 unique individuals, some replicate libraries).
- Normalization: **RPKM scaled to median mapped reads** (per file description). 5 populations: CEU, FIN, GBR, TSI, YRI.
- ⚠ **BUILD FOOTGUN:** the `Coord` column is **GRCh37/hg19**. Do NOT use it. **Join by Ensembl `TargetID`;
  take GRCh38 TSS/coordinates from GENCODE v44.** (Gene IDs are build-independent; may need to strip version suffix.)
- Preprocessing to add (Phase 1/nb01): TMM/inverse-normal per gene, covariate correction (two schemes), z-score.
- Target path: `data/raw/geuvadis/GD660.GeneQuantRPKM.txt.gz`.

## MAGE v1.0 — multi-ancestry LCL RNA-seq (ancestry axis)
- **Zenodo:** `https://zenodo.org/records/10535719` · DOI `10.5281/zenodo.10535719` · **CC-BY-4.0.**
- Paper: Taylor/Battle et al., *Nature* 2024 (`s41586-024-07708-2`); RNA-seq SRA `PRJNA851328`.
- **731 individuals, 26 populations / 5 continental groups.** Processed expression + eQTL/sQTL results included.
- ⚠ **PACKAGING CAVEAT:** Zenodo ships a **single 59.5 GB `MAGE.v1.0.data.zip`**. We only need the normalized
  expression matrix + eQTL calls + sample metadata (a small fraction). Options for later: (a) partial-extract
  specific members from the zip via HTTP range + `zip` central directory, or (b) the Dropbox mirror with
  individual files (linked from the GitHub repo `mccoy-lab/MAGE`). **Resolve before the ancestry-axis phase.**
- **Inventory only for Phase 0** — not needed for the smoke test.

## GENCODE v44 — gene/transcript/TSS annotation (GRCh38)
- **URL:** `http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz` (**50 MB**, HTTP 200).
- Provides canonical/MANE transcript TSS for window anchoring; source of GRCh38 coords to override GEUVADIS hg19 Coord.
- Target path: `data/raw/gencode/gencode.v44.annotation.gtf.gz`.

## Reference FASTA — Ensembl GRCh38 (release-110)
- **Full primary assembly:** `.../Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz` (**882 MB**).
- **Per-chromosome (smoke test):** `.../Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz` (~12 MB) — use for Phase 0.
- ⚠ Ensembl chr naming = **`22`** (no `chr`); 1000G uses **`chr22`**. Handle the mismatch when cross-referencing.
- Target path: `data/raw/reference/`.

## GTEx v8 — ⚠ FLAGGED, NOT A CORE DEPENDENCY
- Expression + eQTLs are open; **genotypes are dbGaP controlled-access.** Per rule 11 / kickoff: do **not**
  build the pipeline on GTEx genotypes unless the user confirms dbGaP approval. Use only for external
  eQTL direction-of-effect validation or a cross-tissue robustness note.

---

## Smoke-test-minimal fetch set (Phase 0 only; ≈123 MB total, all to /ephemeral)
1. Ensembl GRCh38 **chr22** FASTA (~12 MB)
2. GENCODE v44 GTF (50 MB)
3. GEUVADIS GD660 matrix (61 MB)
4. Enformer pretrained weights (~1 GB, auto from HuggingFace `EleutherAI/enformer-official-rough`)
5. 1000G chr22 — **streamed window only, 0 MB on disk**

Disk after: well within `/ephemeral` 700 G. Deferred (need approval before Phase 1): full reference,
MAGE 59.5 GB slice, kinship/pedigree, per-chromosome eQTL call sets.
