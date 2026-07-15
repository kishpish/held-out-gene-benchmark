"""Train/val/test splits + leakage audit (METHODOLOGY §4, hard rules 3 & 5).

Three independent split axes, combined at evaluation time into the task-cell grid (§1.1):
  - GENE axis: by CHROMOSOME, never random (removes LD / shared-enhancer / TAD leakage, §4.1).
  - INDIVIDUAL axis: disjoint, with RELATED individuals kept on the same side via FamilyID (§4.2, rule 5),
    stratified by population so ancestry composition is balanced across splits.
  - ANCESTRY axis: EUR train ↔ AFR test (and reverse) for the portability stress cell (§4.3).

Everything is deterministic (seeded) and serialized to frozen JSON specs under splits/.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Primary chromosome fold (Enformer/Borzoi convention). Rotations for robustness in §10.
CHROM_FOLDS = {
    "fold0": {"train": [f"chr{i}" for i in range(1, 19)],
              "val": ["chr19", "chr20"], "test": ["chr21", "chr22"]},
    "fold1": {"train": [f"chr{i}" for i in [1, 2, 3, 4] + list(range(9, 23))],
              "val": ["chr7", "chr8"], "test": ["chr5", "chr6"]},
    "fold2": {"train": [f"chr{i}" for i in range(5, 23)],
              "val": ["chr3", "chr4"], "test": ["chr1", "chr2"]},
}


def gene_split(gene_meta: pd.DataFrame, fold: str = "fold0") -> dict[str, list[str]]:
    """Assign gene_ids to train/val/test by chromosome."""
    f = CHROM_FOLDS[fold]
    out = {}
    for part, chroms in f.items():
        out[part] = list(gene_meta.index[gene_meta.chrom.isin(chroms)])
    return out


def individual_split(indiv_meta: pd.DataFrame, seed: int = 0,
                     fracs=(0.70, 0.15, 0.15)) -> dict[str, list[str]]:
    """Relatedness-aware, population-stratified individual split (families kept together)."""
    rng = np.random.default_rng(seed)
    assign: dict[str, str] = {}          # FamilyID -> part
    parts = ["train", "val", "test"]
    # stratify family assignment within each population (use founder's population per family)
    fam_pop = indiv_meta.groupby("FamilyID").Population.agg(lambda s: s.iloc[0])
    for pop, fams in fam_pop.groupby(fam_pop):
        ids = list(fams.index)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(n * fracs[0])); n_va = int(round(n * fracs[1]))
        for i, fam in enumerate(ids):
            assign[fam] = parts[0] if i < n_tr else (parts[1] if i < n_tr + n_va else parts[2])
    out = {p: [] for p in parts}
    for sample, fam in indiv_meta.FamilyID.items():
        out[assign[fam]].append(sample)
    return out


def ancestry_split(indiv_meta: pd.DataFrame, train_super="EUR", test_super="AFR") -> dict[str, list[str]]:
    """Ancestry-holdout individual split for portability (§4.3)."""
    return {
        "train": list(indiv_meta.index[indiv_meta.Superpopulation == train_super]),
        "test": list(indiv_meta.index[indiv_meta.Superpopulation == test_super]),
        "train_superpop": train_super, "test_superpop": test_super,
    }


# ------------------------------- leakage audit -------------------------------
@dataclass
class AuditResult:
    passed: bool
    checks: dict


def audit_gene_split(gene_meta: pd.DataFrame, gsplit: dict,
                     paralog_pairs: set[tuple[str, str]] | None = None) -> AuditResult:
    """No gene on two sides; no chromosome shared; optional paralog leakage between train and test."""
    checks = {}
    train, val, test = set(gsplit["train"]), set(gsplit["val"]), set(gsplit["test"])
    checks["no_gene_overlap"] = (len(train & test) == 0 and len(train & val) == 0 and len(val & test) == 0)
    chr_train = set(gene_meta.loc[list(train), "chrom"]); chr_test = set(gene_meta.loc[list(test), "chrom"])
    checks["no_chrom_overlap_train_test"] = len(chr_train & chr_test) == 0

    if paralog_pairs is not None:
        base = gene_meta.assign(base=gene_meta.index.str.split(".").str[0])["base"]
        train_base = set(base.loc[list(train)]); test_base = set(base.loc[list(test)])
        leaks = [(a, b) for (a, b) in paralog_pairs if a in test_base and b in train_base]
        checks["n_test_genes_with_train_paralog"] = len({a for a, _ in leaks})
        checks["paralog_leak_pairs"] = len(leaks)
    return AuditResult(passed=all(v is True for v in checks.values()
                                  if isinstance(v, bool)), checks=checks)


def audit_individual_split(indiv_meta: pd.DataFrame, isplit: dict) -> AuditResult:
    checks = {}
    tr, va, te = set(isplit["train"]), set(isplit["val"]), set(isplit["test"])
    checks["no_individual_overlap"] = (len(tr & te) == 0 and len(tr & va) == 0 and len(va & te) == 0)
    # no family split across parts
    fam = indiv_meta.FamilyID
    part_of = {s: p for p in ("train", "val", "test") for s in isplit[p]}
    fam_parts = {}
    for s, f in fam.items():
        fam_parts.setdefault(f, set()).add(part_of.get(s))
    checks["no_family_across_splits"] = all(len(v) == 1 for v in fam_parts.values())
    return AuditResult(passed=all(checks.values()), checks=checks)


def save_split(spec: dict, path: Path) -> None:
    path.write_text(json.dumps(spec, indent=2))
