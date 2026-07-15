#!/usr/bin/env python
"""Phase 5a — linear cross-individual baselines on the selected (learnable) genes.

Documented in docs/PHASE5_baselines_and_failure_reproduction.md. Mean / lead-eQTL / elastic-net,
trained on TRAIN individuals and scored on HELD-OUT individuals of the same gene, within EUR (matches
the Phase-3 cis-h² ceiling and removes ancestry as a confound). Establishes the {gene present, held-out
individual} performance the deep method must beat — and the reference for the "PrediXcan is undefined on a
truly held-out gene" argument. No reference genome / GPU needed (genotype streaming only).

Writes results/phase5_linear_baselines.csv + results/phase5_linear_baselines.png.
Run: .venv/bin/python scripts/phase5_linear_baselines.py
"""
import sys
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import PROCESSED, RESULTS, SPLITS, CACHE, kgp_vcf_url
import baselines
from baselines import cis_dosage, evaluate_gene

WORKERS = 6


def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    meds = [np.median(rng.choice(x, len(x), replace=True)) for _ in range(n)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main() -> None:
    gm = pd.read_parquet(PROCESSED / "gene_meta.parquet")
    im = pd.read_parquet(PROCESSED / "individual_meta.parquet")
    target = pd.read_parquet(PROCESSED / "geuvadis_target_int.parquet")
    selected = (RESULTS / "phase3_selected_genes_chr22.txt").read_text().split()
    h2tab = pd.read_csv(RESULTS / "phase3_cis_h2_chr22.csv").set_index("gene_id")
    isplit = json.loads((SPLITS / "individual_split.json").read_text())

    eur = set(im.index[im.Superpopulation == "EUR"])
    train = [s for s in isplit["train"] if s in eur]
    test = [s for s in isplit["test"] if s in eur]
    samples = train + test
    tr_idx = np.arange(len(train)); te_idx = np.arange(len(train), len(samples))
    from personal import write_sample_file
    sample_file = write_sample_file(samples, CACHE / "phase5_eur_split.txt")
    url = kgp_vcf_url("chr22")
    print(f"selected genes: {len(selected)} | EUR train={len(train)} test={len(test)}")

    def run_gene(gid):
        g = gm.loc[gid]
        try:
            X, af, pos = cis_dosage(url, "chr22", int(g.tss), samples, sample_file)
            y = target.loc[gid, samples].to_numpy(float)
            r = evaluate_gene(gid, X, y, tr_idx, te_idx)
            return dict(gene_id=gid, gene=g.gene_name, n_snp=r.n_snp,
                        rho_lead=r.rho_lead, rho_enet=r.rho_enet, rho_mean=r.rho_mean,
                        r_enet=r.r_enet, enet_nonzero=r.enet_nonzero,
                        h2_eur=float(h2tab.loc[gid, "h2_eur"]) if gid in h2tab.index else np.nan)
        except Exception as e:
            return dict(gene_id=gid, gene=g.gene_name, error=str(e)[:80])

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for f in as_completed([ex.submit(run_gene, gid) for gid in selected]):
            rows.append(f.result())
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "phase5_linear_baselines.csv", index=False)
    ok = df.dropna(subset=["rho_enet"]) if "rho_enet" in df else df
    print(f"evaluated {len(ok)}/{len(selected)} genes in {time.time()-t0:.0f}s")

    # ---- summary + bootstrap CIs -----------------------------------------------
    print("\nPer-gene cross-individual Spearman ρ (held-out EUR individuals), median [95% CI]:")
    for col, name in [("rho_mean", "mean       "), ("rho_lead", "lead-eQTL  "), ("rho_enet", "elastic-net")]:
        v = ok[col].to_numpy()
        lo, hi = boot_ci(v)
        fpos = float((v > 0).mean())
        print(f"  {name}  median={np.median(v):+.3f}  [{lo:+.3f}, {hi:+.3f}]  frac(ρ>0)={fpos:.2f}")
    # fraction of cis-h² ceiling explained by elastic-net (ρ² / h²)
    m = ok.dropna(subset=["h2_eur"])
    m = m[m.h2_eur > 0.05]
    frac = np.median(np.clip(m.rho_enet.clip(0) ** 2 / m.h2_eur, 0, 2))
    print(f"\n  elastic-net median ρ²/h²_cis on learnable genes (n={len(m)}): {frac:.2f} "
          f"(fraction of the cis-h² ceiling explained)")

    # ---- figure -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    data = [ok.rho_mean, ok.rho_lead, ok.rho_enet]
    bp = axes[0].boxplot(data, tick_labels=["mean", "lead-eQTL", "elastic-net"], showmeans=True,
                         patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#BBBBBB", "#DD8452", "#4C72B0"]):
        patch.set_facecolor(c); patch.set_alpha(0.8)
    axes[0].axhline(0, color="k", lw=0.8, ls=":")
    axes[0].set_ylabel("per-gene cross-individual Spearman ρ")
    axes[0].set_title(f"Linear baselines on {len(ok)} learnable chr22 genes\n(trained→held-out EUR individuals)")
    axes[1].scatter(np.sqrt(m.h2_eur), m.rho_enet, s=22, alpha=0.75, c="#4C72B0")
    lim = [0, max(1.0, float(np.sqrt(m.h2_eur).max()) + 0.05)]
    axes[1].plot(lim, lim, "k--", lw=1, label="ρ = √h² (ceiling)")
    axes[1].set_xlabel("√h²_cis (correlation ceiling)"); axes[1].set_ylabel("elastic-net ρ")
    axes[1].set_title("Elastic-net vs the cis-h² ceiling"); axes[1].legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(RESULTS / "phase5_linear_baselines.png", dpi=130)
    print("\nwrote results/phase5_linear_baselines.csv, phase5_linear_baselines.png")
    print("Phase 5a linear baselines complete.")


if __name__ == "__main__":
    main()
