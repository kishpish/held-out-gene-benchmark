"""GENCODE GTF parsing → per-gene canonical-transcript TSS (GRCh38).

We anchor model windows on the TSS of the **Ensembl_canonical** transcript (falling back to
MANE_Select, then the gene span). GENCODE uses UCSC-style chr names ('chr22').
"""
from __future__ import annotations
import gzip
import re
from pathlib import Path

import pandas as pd

_ATTR = re.compile(r'(\w+) "([^"]*)"')


def _attrs(field: str) -> dict:
    return dict(_ATTR.findall(field))


AUTOSOMES = [f"chr{i}" for i in range(1, 23)]


def load_genes(gtf_path: Path, chrom: str | list[str] | None = "chr22",
               gene_type: str = "protein_coding") -> pd.DataFrame:
    """Return genes with canonical-transcript TSS.

    `chrom` may be a single name ('chr22'), a list, or None for all autosomes (chr1..chr22).
    Columns: gene_id, gene_id_base, gene_name, chrom, strand, gene_start, gene_end, tss (1-based).
    """
    if chrom is None:
        keep = set(AUTOSOMES)
    elif isinstance(chrom, str):
        keep = {chrom if chrom.startswith("chr") else f"chr{chrom}"}
    else:
        keep = {c if c.startswith("chr") else f"chr{c}" for c in chrom}
    genes: dict[str, dict] = {}
    # canonical TSS candidates per gene: (priority, tss). priority 0=MANE,1=canonical,2=fallback
    tss_pick: dict[str, tuple[int, int]] = {}

    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            c = line.split("\t", 1)[0]
            if c not in keep:
                continue
            f = line.rstrip("\n").split("\t")
            feat, start, end, strand, attr = f[2], int(f[3]), int(f[4]), f[6], f[8]
            a = _attrs(attr)
            if a.get("gene_type") != gene_type:
                continue
            gid = a["gene_id"]
            if feat == "gene":
                genes[gid] = dict(gene_id=gid, gene_id_base=gid.split(".")[0],
                                  gene_name=a.get("gene_name", ""), chrom=c, strand=strand,
                                  gene_start=start, gene_end=end)
            elif feat == "transcript":
                tags = attr  # tags appear as tag "X"
                tss = start if strand == "+" else end
                if "MANE_Select" in tags:
                    prio = 0
                elif "Ensembl_canonical" in tags:
                    prio = 1
                else:
                    prio = 2
                cur = tss_pick.get(gid)
                if cur is None or prio < cur[0]:
                    tss_pick[gid] = (prio, tss)

    rows = []
    for gid, g in genes.items():
        pick = tss_pick.get(gid)
        tss = pick[1] if pick else (g["gene_start"] if g["strand"] == "+" else g["gene_end"])
        rows.append({**g, "tss": tss})
    df = pd.DataFrame(rows).sort_values("gene_start").reset_index(drop=True)
    return df


if __name__ == "__main__":
    from paths import GENCODE_DIR
    df = load_genes(GENCODE_DIR / "gencode.v44.annotation.gtf.gz", "chr22")
    print(f"{len(df)} protein-coding genes on chr22")
    print(df.head(3).to_string())
