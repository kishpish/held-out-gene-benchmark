#!/usr/bin/env python
"""Phase 4 — build frozen train/val/test splits + leakage audit (METHODOLOGY §4).

Documented in docs/PHASE4_splits_and_leakage_audit.md. Produces (in splits/):
  - gene_split_fold{0,1,2}.json     chromosome-level gene splits (+ rotations)
  - individual_split.json           relatedness-aware, population-stratified individual split
  - ancestry_split.json             EUR train / AFR test (portability)
  - task_cells.json                 sizes of the evaluation grid cells (§1.1)
  - leakage_audit.md                the archived audit report (§4.4)

Homology audit uses the cached Ensembl BioMart paralog table (base gene ids).
Run: .venv/bin/python scripts/phase4_build_splits.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd

from paths import PROCESSED, SPLITS, RAW
import splits as S

PARALOG_TSV = RAW / "meta" / "ensembl_paralogs_autosomes.tsv"


def load_paralog_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, sep="\t", header=None, names=["a", "b"]).dropna()
    return set(zip(df.a, df.b))          # unversioned base ids (both directions present)


def main() -> None:
    SPLITS.mkdir(exist_ok=True)
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    paralogs = load_paralog_pairs(PARALOG_TSV)
    print(f"genes={len(gm)}  individuals={len(im)}  paralog_pairs={len(paralogs):,}")

    # ---- individual + ancestry splits (fold-independent) ------------------------
    isplit = S.individual_split(im, seed=0)
    asplit = S.ancestry_split(im, "EUR", "AFR")
    ia = S.audit_individual_split(im, isplit)
    print("individual split:", {k: len(v) for k, v in isplit.items()},
          "| audit:", ia.checks, "PASS" if ia.passed else "FAIL")
    S.save_split({**isplit, "seed": 0, "audit": ia.checks}, SPLITS / "individual_split.json")
    S.save_split(asplit, SPLITS / "ancestry_split.json")

    # ---- gene splits per fold + homology audit ----------------------------------
    audit_md = ["# Phase 4 — Leakage Audit Report", "",
                f"- genes: {len(gm)} | individuals: {len(im)} | paralog pairs: {len(paralogs):,}",
                f"- individual split (families grouped, pop-stratified): "
                f"train={len(isplit['train'])}, val={len(isplit['val'])}, test={len(isplit['test'])}; "
                f"audit {'PASS' if ia.passed else 'FAIL'} {ia.checks}", ""]
    task_cells = {}
    for fold in ["fold0", "fold1", "fold2"]:
        gsplit = S.gene_split(gm, fold)
        ga = S.audit_gene_split(gm, gsplit, paralogs)
        S.save_split({**gsplit, "fold": fold, "audit": ga.checks}, SPLITS / f"gene_split_{fold}.json")

        # homology-hard test subset: a test gene is "hard" if none of its paralogs is a train gene
        base = gm.assign(base=gm.index.str.split(".").str[0])["base"]
        train_base = set(base.loc[gsplit["train"]]); test_ids = gsplit["test"]
        test_base = {g: base[g] for g in test_ids}
        para_of = {}
        for a, b in paralogs:
            para_of.setdefault(a, set()).add(b)
        hard = [g for g in test_ids if not (para_of.get(test_base[g], set()) & train_base)]

        print(f"[{fold}] genes train/val/test = "
              f"{len(gsplit['train'])}/{len(gsplit['val'])}/{len(gsplit['test'])} | "
              f"paralog audit: {ga.checks} | homology-hard test genes: {len(hard)}/{len(test_ids)}")
        audit_md += [f"## Gene split {fold}",
                     f"- chromosomes: train={S.CHROM_FOLDS[fold]['train']}",
                     f"  val={S.CHROM_FOLDS[fold]['val']} test={S.CHROM_FOLDS[fold]['test']}",
                     f"- sizes train/val/test = {len(gsplit['train'])}/{len(gsplit['val'])}/{len(gsplit['test'])}",
                     f"- no gene overlap: {ga.checks['no_gene_overlap']}; "
                     f"no chrom overlap (train/test): {ga.checks['no_chrom_overlap_train_test']}",
                     f"- test genes with ≥1 train paralog: "
                     f"{ga.checks['n_test_genes_with_train_paralog']} "
                     f"({ga.checks['paralog_leak_pairs']} pairs) → **homology-hard test subset "
                     f"= {len(hard)}/{len(test_ids)} genes** (report metrics on this subset too)", ""]

        # main task cell (§1.1): held-out gene × held-out individual (fold0, in-train ancestry)
        if fold == "fold0":
            task_cells = {
                "reproduction__seenGene_heldIndiv": {
                    "genes": len(gsplit["train"]), "individuals": len(isplit["test"])},
                "CORE__heldGene_heldIndiv": {
                    "genes": len(gsplit["test"]), "individuals": len(isplit["test"]),
                    "homology_hard_genes": len(hard)},
                "STRETCH__heldGene_heldIndiv_heldAncestry": {
                    "genes": len(gsplit["test"]), "individuals": len(asplit["test"])},
                "val__heldGene_heldIndiv": {
                    "genes": len(gsplit["val"]), "individuals": len(isplit["val"])},
            }

    S.save_split(task_cells, SPLITS / "task_cells.json")
    audit_md += ["## Task-cell grid (fold0)", "```json", json.dumps(task_cells, indent=2), "```"]
    (SPLITS / "leakage_audit.md").write_text("\n".join(audit_md))
    print("\ntask cells:", json.dumps(task_cells, indent=2))
    print("wrote splits/: gene_split_fold{0,1,2}.json, individual_split.json, ancestry_split.json, "
          "task_cells.json, leakage_audit.md")
    print("Phase 4 splits complete.")


if __name__ == "__main__":
    main()
