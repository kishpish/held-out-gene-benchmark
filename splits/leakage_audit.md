# Phase 4 — Leakage Audit Report

- genes: 12360 | individuals: 449 | paralog pairs: 251,275
- individual split (families grouped, pop-stratified): train=309, val=71, test=69; audit PASS {'no_individual_overlap': True, 'no_family_across_splits': True}

## Gene split fold0
- chromosomes: train=['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18']
  val=['chr19', 'chr20'] test=['chr21', 'chr22']
- sizes train/val/test = 10610/1294/456
- no gene overlap: True; no chrom overlap (train/test): True
- test genes with ≥1 train paralog: 300 (1914 pairs) → **homology-hard test subset = 156/456 genes** (report metrics on this subset too)

## Gene split fold1
- chromosomes: train=['chr1', 'chr2', 'chr3', 'chr4', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
  val=['chr7', 'chr8'] test=['chr5', 'chr6']
- sizes train/val/test = 10108/1012/1240
- no gene overlap: True; no chrom overlap (train/test): True
- test genes with ≥1 train paralog: 819 (5997 pairs) → **homology-hard test subset = 421/1240 genes** (report metrics on this subset too)

## Gene split fold2
- chromosomes: train=['chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22']
  val=['chr3', 'chr4'] test=['chr1', 'chr2']
- sizes train/val/test = 9054/1148/2158
- no gene overlap: True; no chrom overlap (train/test): True
- test genes with ≥1 train paralog: 1424 (9049 pairs) → **homology-hard test subset = 734/2158 genes** (report metrics on this subset too)

## Task-cell grid (fold0)
```json
{
  "reproduction__seenGene_heldIndiv": {
    "genes": 10610,
    "individuals": 69
  },
  "CORE__heldGene_heldIndiv": {
    "genes": 456,
    "individuals": 69,
    "homology_hard_genes": 156
  },
  "STRETCH__heldGene_heldIndiv_heldAncestry": {
    "genes": 456,
    "individuals": 89
  },
  "val__heldGene_heldIndiv": {
    "genes": 1294,
    "individuals": 71
  }
}
```