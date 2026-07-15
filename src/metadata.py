"""1000 Genomes sample metadata (population, super-population, pedigree).

Loads the NYGC 3,202-sample pedigree+population file. Provides ancestry labels for the
ancestry axis (METHODOLOGY §4.3) and parent IDs / family IDs for relatedness-aware splitting
(hard rule 5 / §4.2). Small file, downloaded to data/raw/meta/.
"""
from __future__ import annotations
from pathlib import Path

import pandas as pd

from paths import RAW

PED_FILE = RAW / "meta" / "1000G_3202_ped_population.txt"


def load_pedigree(path: Path = PED_FILE) -> pd.DataFrame:
    """Return per-sample metadata.

    Columns: FamilyID, SampleID, FatherID, MotherID, Sex, Population, Superpopulation,
    plus `is_founder` (both parents == '0') for relatedness grouping.
    """
    df = pd.read_csv(path, sep=r"\s+")
    df["is_founder"] = (df.FatherID == "0") & (df.MotherID == "0")
    return df


def ancestry_map(path: Path = PED_FILE) -> pd.DataFrame:
    """SampleID -> Population, Superpopulation (indexed by SampleID)."""
    df = load_pedigree(path)
    return df.set_index("SampleID")[["Population", "Superpopulation", "FamilyID",
                                     "FatherID", "MotherID", "is_founder"]]


if __name__ == "__main__":
    df = load_pedigree()
    print(f"{len(df)} samples")
    print(df.Superpopulation.value_counts().to_string())
    print("founders:", int(df.is_founder.sum()), "non-founders:", int((~df.is_founder).sum()))
